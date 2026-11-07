"""
Schema editor for DuckDB backend.
"""

import copy

from django.apps.registry import Apps
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.backends.ddl_references import Statement
from django.db.backends.utils import strip_quotes


class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):
    sql_create_column_inline_fk = "REFERENCES %(to_table)s (%(to_column)s)%(on_delete)s"
    sql_create_unique = "CREATE UNIQUE INDEX %(name)s ON %(table)s (%(columns)s)"
    sql_delete_unique = "DROP INDEX IF EXISTS %(name)s"
    sql_alter_table_comment = None
    sql_alter_column_comment = None

    def quote_value(self, value):
        """Quote a value for use in SQL."""
        if isinstance(value, bool):
            return str(int(value))
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            return "'%s'" % value.replace("'", "''")
        elif value is None:
            return "NULL"
        elif isinstance(value, (bytes, bytearray, memoryview)):
            return "X'%s'" % value.hex()
        else:
            return str(value)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            for sql in self.deferred_sql:
                # Skip deferred FK creation — DuckDB doesn't support ALTER TABLE ADD CONSTRAINT FK
                if (
                    isinstance(sql, Statement)
                    and sql.template
                    and "FOREIGN KEY" in sql.template
                ):
                    continue
                self.execute(sql, None)
        if self.atomic_migration:
            self.atomic.__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def _sequence_name(db_table, column):
        # Must stay in sync with introspection if you implement get_sequences()
        return f"seq_{db_table}_{column}"

    def _is_auto(self, field):
        return field.get_internal_type() in (
            "AutoField",
            "BigAutoField",
            "SmallAutoField",
        )

    def table_sql(self, model, *args, **kwargs):
        """Create sequences for AutoFields before creating the table."""
        for field in model._meta.local_concrete_fields:
            if self._is_auto(field):
                seq_name = self._sequence_name(model._meta.db_table, field.column)
                self.execute(
                    f"CREATE SEQUENCE IF NOT EXISTS {self.quote_name(seq_name)}"
                )
        return super().table_sql(model, *args, **kwargs)

    def _iter_column_sql(
        self, column_db_type, params, model, field, field_db_params, include_default
    ):
        """Override to handle DuckDB auto-increment via sequences."""
        if self._is_auto(field):
            seq_name = self._sequence_name(model._meta.db_table, field.column)
            yield column_db_type
            yield f"DEFAULT nextval('{self.quote_name(seq_name)}')"
            if field.primary_key:
                yield "PRIMARY KEY"
            return

        yield from super()._iter_column_sql(
            column_db_type, params, model, field, field_db_params, include_default
        )

    def delete_model(self, model, *args, **kwargs):
        """Drop sequences when deleting a model."""
        for field in model._meta.local_concrete_fields:
            if self._is_auto(field):
                seq_name = self._sequence_name(model._meta.db_table, field.column)
                self.execute(f"DROP SEQUENCE IF EXISTS {self.quote_name(seq_name)}")
        super().delete_model(model, *args, **kwargs)

    def add_field(self, model, field):
        """Add a field to a model.

        DuckDB doesn't support ALTER TABLE ADD COLUMN with NOT NULL or FK,
        so we recreate the table for those cases.
        """
        # Special case: M2M tables
        if field.many_to_many and field.remote_field.through._meta.auto_created:
            self.create_model(field.remote_field.through)
            return

        # If the field has a FK constraint, we need to recreate the table
        if field.remote_field and field.db_constraint:
            self._remake_table(model, create_field=field)
            return

        # If NOT NULL without default, need to recreate
        if not field.null and self.effective_default(field) is None:
            self._remake_table(model, create_field=field)
            return

        # Otherwise use the standard path
        super().add_field(model, field)

    def _alter_field(
        self,
        model,
        old_field,
        new_field,
        old_type,
        new_type,
        old_db_params,
        new_db_params,
        strict=False,
    ):
        """Defer to table remake only for ALTERs DuckDB cannot perform."""

        old_fks = (old_field.remote_field is not None and old_field.db_constraint) or (
            new_field.remote_field is not None and new_field.db_constraint
        )

        old_internal = old_field.get_internal_type()
        new_internal = new_field.get_internal_type()
        auto_involved = {"AutoField", "BigAutoField", "SmallAutoField"}
        unsupported = (
            old_internal in auto_involved
            or new_internal in auto_involved
            or old_fks
            or (old_field.null and not new_field.null)
        )

        if unsupported:
            self._remake_table(model, alter_fields=[(old_field, new_field)])
            return

        super()._alter_field(
            model,
            old_field,
            new_field,
            old_type,
            new_type,
            old_db_params,
            new_db_params,
            strict,
        )

    def _remake_table(
        self, model, create_field=None, delete_field=None, alter_fields=None
    ):
        """Recreate a table with the updated schema.

        DuckDB doesn't support many ALTER TABLE operations, so we need to
        recreate the table: create a new table, copy data, drop old, rename new.
        """
        alter_fields = alter_fields or []

        # Build the new fields dict
        body = {f.name: f.clone() for f in model._meta.local_concrete_fields}

        # Mapping for data copy
        mapping = {
            f.column: self.quote_name(f.column)
            for f in model._meta.local_concrete_fields
        }

        # Add created field
        if create_field:
            body[create_field.name] = create_field
            if (
                not create_field.has_db_default()
                and not create_field.generated
                and create_field.concrete
            ):
                mapping[create_field.column] = self.prepare_default(
                    self.effective_default(create_field)
                )

        # Remove deleted field
        if delete_field:
            body.pop(delete_field.name, None)
            mapping.pop(delete_field.column, None)

        # Handle altered fields
        for old_field, new_field in alter_fields:
            body.pop(old_field.name, None)
            mapping.pop(old_field.column, None)
            body[new_field.name] = new_field
            mapping[new_field.column] = self.quote_name(old_field.column)
            if old_field.column != new_field.column and self._is_auto(new_field):
                old_seq = self._sequence_name(model._meta.db_table, old_field.column)
                self.execute(f"DROP SEQUENCE IF EXISTS {self.quote_name(old_seq)}")

        # Work inside a new app registry
        apps = Apps()

        # Register related models that FK fields point to, so they can be resolved
        for field in model._meta.local_concrete_fields:
            if field.remote_field and hasattr(field.remote_field, "model"):
                related_model = field.remote_field.model
                if hasattr(related_model, "_meta"):
                    # Create a stub of the related model in the isolated apps
                    if not apps.is_installed(related_model._meta.app_label):
                        related_body = {
                            f.name: f.clone()
                            for f in related_model._meta.local_concrete_fields
                        }
                        related_meta = {
                            "app_label": related_model._meta.app_label,
                            "db_table": related_model._meta.db_table,
                            "apps": apps,
                        }
                        related_meta_obj = type("Meta", (), related_meta)
                        related_body["Meta"] = related_meta_obj
                        related_body["__module__"] = related_model.__module__
                        type(
                            related_model._meta.object_name,
                            related_model.__bases__,
                            related_body,
                        )

        # Construct a model with the original table name for FK resolution
        body_copy = copy.deepcopy(body)
        meta_contents = {
            "app_label": model._meta.app_label,
            "db_table": model._meta.db_table,
            "unique_together": model._meta.unique_together,
            "indexes": model._meta.indexes,
            "constraints": model._meta.constraints,
            "apps": apps,
        }
        meta = type("Meta", (), meta_contents)
        body_copy["Meta"] = meta
        body_copy["__module__"] = model.__module__
        type(model._meta.object_name, model.__bases__, body_copy)

        # Construct a model with a renamed table name
        body_copy = copy.deepcopy(body)
        meta_contents = {
            "app_label": model._meta.app_label,
            "db_table": "new__%s" % strip_quotes(model._meta.db_table),
            "unique_together": model._meta.unique_together,
            "indexes": model._meta.indexes,
            "constraints": model._meta.constraints,
            "apps": apps,
        }
        meta = type("Meta", (), meta_contents)
        body_copy["Meta"] = meta
        body_copy["__module__"] = model.__module__
        new_model = type("New%s" % model._meta.object_name, model.__bases__, body_copy)

        # Create the new table
        self.create_model(new_model)

        # Copy data
        self.execute(
            "INSERT INTO %s (%s) SELECT %s FROM %s"
            % (
                self.quote_name(new_model._meta.db_table),
                ", ".join(self.quote_name(x) for x in mapping),
                ", ".join(mapping.values()),
                self.quote_name(model._meta.db_table),
            )
        )

        # Drop the old table
        self.execute(f"DROP TABLE {self.quote_name(model._meta.db_table)}")

        # Rename the new table
        self.execute(
            f"ALTER TABLE {self.quote_name(new_model._meta.db_table)} "
            f"RENAME TO {self.quote_name(model._meta.db_table)}"
        )

        # Clear deferred SQL that references the temporary table
        self.deferred_sql = [
            s
            for s in self.deferred_sql
            if new_model._meta.db_table not in str(getattr(s, "template", ""))
        ]

    def alter_db_table(
        self, model, old_db_table, new_db_table, old_fields=None, fields=None
    ):
        super().alter_db_table(model, old_db_table, new_db_table, old_fields, fields)
        if fields is not None:
            for field in fields:
                if self._is_auto(field):
                    old_seq = self._sequence_name(old_db_table, field.column)
                    new_seq = self._sequence_name(new_db_table, field.column)
                    self.execute(
                        f"ALTER SEQUENCE {self.quote_name(old_seq)} "
                        f"RENAME TO {self.quote_name(new_seq)}"
                    )
