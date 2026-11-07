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

    def prepare_default(self, value):
        return self.quote_value(value)

    def delete_model(self, model, handle_autom2m=True):
        """Drop sequences when deleting a model."""
        if handle_autom2m:
            for field in model._meta.local_concrete_fields:
                if self._is_auto(field):
                    seq_name = self._sequence_name(model._meta.db_table, field.column)
                    self.execute(f"DROP SEQUENCE IF EXISTS {self.quote_name(seq_name)}")
            super().delete_model(model)
        else:
            self.execute(f"DROP TABLE {self.quote_name(model._meta.db_table)}")
            # Remove all deferred statements referencing the deleted table.
            for sql in list(self.deferred_sql):
                if isinstance(sql, Statement) and sql.references_table(
                    model._meta.db_table
                ):
                    self.deferred_sql.remove(sql)

    def add_field(self, model, field):
        """Add a field to a model.

        DuckDB doesn't support ALTER TABLE ADD COLUMN with NOT NULL or FK,
        so we recreate the table for those cases.
        """
        # Special case: M2M tables
        if field.many_to_many and field.remote_field.through._meta.auto_created:
            self.create_model(field.remote_field.through)
            return

        # If primary key, unique, not null, or FK constraint, need to recreate
        if (
            field.primary_key
            or field.unique
            or not field.null
            or (field.remote_field and field.db_constraint)
            or self.effective_default(field) is not None
        ):
            self._remake_table(model, create_field=field)
            return

        # Otherwise use the standard path
        super().add_field(model, field)

    def remove_field(self, model, field):
        """Remove a field from a model."""
        # M2M fields are a special case
        if field.many_to_many:
            if field.remote_field.through._meta.auto_created:
                self.delete_model(field.remote_field.through)
        else:
            if field.db_parameters(connection=self.connection)["type"] is None:
                return
            self._remake_table(model, delete_field=field)

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
        """Perform a field update by remaking the table."""
        self._remake_table(model, alter_fields=[(old_field, new_field)])

    def _remake_table(
        self, model, create_field=None, delete_field=None, alter_fields=None
    ):
        """Recreate a table with the updated schema.

        DuckDB doesn't support many ALTER TABLE operations, so we need to
        recreate the table: create a new table, copy data, drop old, rename new.
        """
        def is_self_referential(f):
            return f.is_relation and f.remote_field.model is model

        # Build the new fields dict
        body = {
            f.name: f.clone() if is_self_referential(f) else f
            for f in model._meta.local_concrete_fields
        }

        # Mapping for data copy
        mapping = {
            f.column: self.quote_name(f.column)
            for f in model._meta.local_concrete_fields
            if getattr(f, "generated", False) is False
        }
        rename_mapping = {}
        restore_pk_field = None
        alter_fields = alter_fields or []

        if getattr(create_field, "primary_key", False) or any(
            getattr(new_f, "primary_key", False) for _, new_f in alter_fields
        ):
            for name, field in list(body.items()):
                if field.primary_key and not any(
                    name == new_f.name for _, new_f in alter_fields
                ):
                    field.primary_key = False
                    restore_pk_field = field
                    if field.auto_created:
                        del body[name]
                        del mapping[field.column]

        # Add created field
        if create_field:
            body[create_field.name] = create_field
            if (
                not create_field.has_db_default()
                and not getattr(create_field, "generated", False)
                and create_field.concrete
            ):
                mapping[create_field.column] = self.prepare_default(
                    self.effective_default(create_field)
                )

        # Handle altered fields
        for old_field, new_field in alter_fields:
            body.pop(old_field.name, None)
            mapping.pop(old_field.column, None)
            body[new_field.name] = new_field
            rename_mapping[old_field.name] = new_field.name
            if getattr(new_field, "generated", False):
                continue
            if old_field.null and not new_field.null:
                if not new_field.has_db_default():
                    default = self.prepare_default(self.effective_default(new_field))
                else:
                    default, _ = self.db_default_sql(new_field)
                case_sql = "coalesce(%(col)s, %(default)s)" % {
                    "col": self.quote_name(old_field.column),
                    "default": default,
                }
                mapping[new_field.column] = case_sql
            else:
                mapping[new_field.column] = self.quote_name(old_field.column)
            if old_field.column != new_field.column and self._is_auto(new_field):
                old_seq = self._sequence_name(model._meta.db_table, old_field.column)
                self.execute(f"DROP SEQUENCE IF EXISTS {self.quote_name(old_seq)}")

        # Remove deleted field
        if delete_field:
            body.pop(delete_field.name, None)
            mapping.pop(delete_field.column, None)
            if self._is_auto(delete_field):
                seq = self._sequence_name(model._meta.db_table, delete_field.column)
                self.execute(f"DROP SEQUENCE IF EXISTS {self.quote_name(seq)}")
            if (
                delete_field.many_to_many
                and delete_field.remote_field.through._meta.auto_created
            ):
                return self.delete_model(delete_field.remote_field.through)

        # Work inside a new app registry
        apps = Apps()

        # Work out unique_together, indexes, constraints
        unique_together = [
            [rename_mapping.get(n, n) for n in unique]
            for unique in model._meta.unique_together
        ]
        indexes = list(model._meta.indexes)
        if delete_field:
            indexes = [
                index for index in indexes if delete_field.name not in index.fields
            ]
        constraints = list(model._meta.constraints)

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
            "unique_together": unique_together,
            "indexes": indexes,
            "constraints": constraints,
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
            "unique_together": unique_together,
            "indexes": indexes,
            "constraints": constraints,
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

        auto_fields = [
            field
            for field in new_model._meta.local_concrete_fields
            if self._is_auto(field)
        ]

        # Delete the old table to make way for the new
        self.delete_model(model, handle_autom2m=False)

        # Recreate auto-field sequences after copying explicit primary keys.
        for field in auto_fields:
            temp_seq = self._sequence_name(new_model._meta.db_table, field.column)
            seq = self._sequence_name(model._meta.db_table, field.column)
            with self.connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT COALESCE(MAX({self.quote_name(field.column)}), 0) + 1 "
                    f"FROM {self.quote_name(new_model._meta.db_table)}"
                )
                next_value = cursor.fetchone()[0]
            self.execute(
                f"ALTER TABLE {self.quote_name(new_model._meta.db_table)} "
                f"ALTER COLUMN {self.quote_name(field.column)} DROP DEFAULT"
            )
            self.execute(f"DROP SEQUENCE IF EXISTS {self.quote_name(temp_seq)}")
            self.execute(f"DROP SEQUENCE IF EXISTS {self.quote_name(seq)}")
            self.execute(
                f"CREATE SEQUENCE {self.quote_name(seq)} START WITH {next_value}"
            )
            self.execute(
                f"ALTER TABLE {self.quote_name(new_model._meta.db_table)} "
                f"ALTER COLUMN {self.quote_name(field.column)} SET DEFAULT "
                f"nextval('{self.quote_name(seq)}')"
            )

        # Rename the new table to the original table name
        super().alter_db_table(
            new_model,
            new_model._meta.db_table,
            model._meta.db_table,
        )

        # Run deferred SQL on correct table
        for sql in self.deferred_sql:
            if (
                isinstance(sql, Statement)
                and sql.template
                and "FOREIGN KEY" in sql.template
            ):
                continue
            self.execute(sql)
        self.deferred_sql = []

        if restore_pk_field:
            restore_pk_field.primary_key = True

    def alter_db_table(
        self, model, old_db_table, new_db_table
    ):
        super().alter_db_table(model, old_db_table, new_db_table)
        for field in model._meta.local_concrete_fields:
            if self._is_auto(field):
                old_seq = self._sequence_name(old_db_table, field.column)
                new_seq = self._sequence_name(new_db_table, field.column)
                with self.connection.cursor() as cursor:
                    cursor.execute(
                        f"SELECT COALESCE(MAX({self.quote_name(field.column)}), 0) + 1 "
                        f"FROM {self.quote_name(new_db_table)}"
                    )
                    next_val = cursor.fetchone()[0]
                self.execute(f"CREATE SEQUENCE IF NOT EXISTS {self.quote_name(new_seq)} START WITH {next_val}")
                self.execute(
                    f"ALTER TABLE {self.quote_name(new_db_table)} "
                    f"ALTER COLUMN {self.quote_name(field.column)} SET DEFAULT "
                    f"nextval('{self.quote_name(new_seq)}')"
                )
                self.execute(f"DROP SEQUENCE IF EXISTS {self.quote_name(old_seq)}")
