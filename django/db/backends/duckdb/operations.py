"""
Database operations for DuckDB backend.
"""

import datetime
import decimal
import uuid
from collections.abc import Mapping

from django.conf import settings
from django.db import DatabaseError
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.models.constants import OnConflict
from django.db.models.expressions import Col
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime, parse_time


class DatabaseOperations(BaseDatabaseOperations):
    cast_char_field_without_max_length = "VARCHAR"
    cast_data_types = {
        "DateField": "VARCHAR",
        "DateTimeField": "VARCHAR",
    }
    explain_prefix = "EXPLAIN"

    def date_extract_sql(self, lookup_type, sql, params):
        return f"EXTRACT({lookup_type.lower()} FROM {sql})", params

    def date_trunc_sql(self, lookup_type, sql, params, tzname=None):
        return f"DATE_TRUNC('{lookup_type.lower()}', {sql})", params

    def time_trunc_sql(self, lookup_type, sql, params, tzname=None):
        return f"TIME_TRUNC('{lookup_type.lower()}', {sql})", params

    def datetime_trunc_sql(self, lookup_type, sql, params, tzname=None):
        return f"TIMESTAMP_TRUNC('{lookup_type.lower()}', {sql})", params

    def pk_default_value(self):
        return "NULL"

    def last_executed_query(self, cursor, sql, params):
        if not params:
            return sql

        def _literal(v):
            if v is None:
                return "NULL"
            if isinstance(v, bool):
                return "1" if v else "0"
            if isinstance(v, (int, float, decimal.Decimal)):
                return str(v)
            if isinstance(v, (bytes, bytearray, memoryview)):
                return "X'%s'" % bytes(v).hex()
            return "'%s'" % str(v).replace("'", "''")

        if isinstance(params, Mapping):
            values = {k: _literal(v) for k, v in params.items()}
            return sql % values
        return sql % tuple(_literal(p) for p in params)

    def quote_name(self, name):
        if name.startswith('"') and name.endswith('"'):
            return name
        return '"%s"' % name

    def no_limit_value(self):
        return None

    def sql_flush(self, style, tables, *, reset_sequences=False, allow_cascade=False):
        sql = [
            "%s %s %s;"
            % (
                style.SQL_KEYWORD("DELETE"),
                style.SQL_KEYWORD("FROM"),
                style.SQL_FIELD(self.quote_name(table)),
            )
            for table in tables
        ]
        if reset_sequences and tables:
            sql.extend(
                self.sequence_reset_by_name_sql(style, [{"table": t} for t in tables])
            )
        return sql

    def sequence_reset_by_name_sql(self, style, sequences):
        """
        Reset sequences for the given tables.

        Sequences are created by the schema editor for auto fields
        (seq_<table>_<column>, see DatabaseSchemaEditor._sequence_name).
        DuckDB has no sequence RESTART command, so drop and recreate each
        sequence starting after the current MAX(pk).
        """
        sql = []
        with self.connection.cursor() as cursor:
            introspection = self.connection.introspection
            existing = {
                row[0]
                for row in cursor.execute(
                    "SELECT sequence_name FROM duckdb_sequences() "
                    "WHERE schema_name = 'main'"
                ).fetchall()
            }
            for sequence in sequences:
                table = sequence["table"]
                pk_columns = introspection.get_primary_key_columns(cursor, table)
                if not pk_columns:
                    continue
                column = sequence.get("column") or pk_columns[0]
                seq_name = f"seq_{table}_{column}"
                if seq_name not in existing:
                    continue
                sql.append("DROP SEQUENCE IF EXISTS %s" % self.quote_name(seq_name))
                sql.append(
                    "CREATE SEQUENCE %s START WITH "
                    "(SELECT COALESCE(MAX(%s), 0) + 1 FROM %s)"
                    % (
                        self.quote_name(seq_name),
                        self.quote_name(column),
                        self.quote_name(table),
                    )
                )
        return sql

    def adapt_datetimefield_value(self, value):
        if value is None:
            return None
        if timezone.is_aware(value):
            if settings.USE_TZ:
                value = timezone.make_naive(value, self.connection.timezone)
            else:
                raise ValueError(
                    "DuckDB backend does not support timezone-aware datetimes when "
                    "USE_TZ is False."
                )
        return value

    def adapt_timefield_value(self, value):
        if value is None:
            return None
        if timezone.is_aware(value):
            raise ValueError("DuckDB backend does not support timezone-aware times.")
        return value

    def get_db_converters(self, expression):
        converters = super().get_db_converters(expression)
        internal_type = expression.output_field.get_internal_type()
        if internal_type == "DateTimeField":
            converters.append(self.convert_datetimefield_value)
        elif internal_type == "DateField":
            converters.append(self.convert_datefield_value)
        elif internal_type == "TimeField":
            converters.append(self.convert_timefield_value)
        elif internal_type == "DecimalField":
            converters.append(self.get_decimalfield_converter(expression))
        elif internal_type == "UUIDField":
            converters.append(self.convert_uuidfield_value)
        elif internal_type == "BooleanField":
            converters.append(self.convert_booleanfield_value)
        return converters

    def convert_datetimefield_value(self, value, expression, connection):
        if value is not None:
            if not isinstance(value, datetime.datetime):
                value = parse_datetime(value)
            if settings.USE_TZ and not timezone.is_aware(value):
                value = timezone.make_aware(value, self.connection.timezone)
        return value

    def convert_datefield_value(self, value, expression, connection):
        if value is not None:
            if not isinstance(value, datetime.date):
                value = parse_date(value)
        return value

    def convert_timefield_value(self, value, expression, connection):
        if value is not None:
            if not isinstance(value, datetime.time):
                value = parse_time(value)
        return value

    @staticmethod
    def _create_decimal(value):
        if isinstance(value, (int, str)):
            return decimal.Decimal(value)
        return decimal.Context(prec=15).create_decimal_from_float(value)

    def get_decimalfield_converter(self, expression):
        if (
            isinstance(expression, Col)
            and expression.output_field.decimal_places is not None
        ):
            quantize_value = decimal.Decimal(1).scaleb(
                -expression.output_field.decimal_places
            )

            def converter(value, expression, connection):
                if value is not None:
                    return self._create_decimal(value).quantize(
                        quantize_value, context=expression.output_field.context
                    )

        else:

            def converter(value, expression, connection):
                if value is not None:
                    return self._create_decimal(value)

        return converter

    def convert_uuidfield_value(self, value, expression, connection):
        if value is not None:
            value = uuid.UUID(value)
        return value

    def convert_booleanfield_value(self, value, expression, connection):
        return bool(value) if value in (1, 0) else value

    def combine_expression(self, connector, sub_expressions):
        if connector == "^":
            return "POWER(%s)" % ",".join(sub_expressions)
        elif connector == "#":
            return "BITXOR(%s)" % ",".join(sub_expressions)
        return super().combine_expression(connector, sub_expressions)

    def combine_duration_expression(self, connector, sub_expressions):
        if connector not in ["+", "-", "*", "/"]:
            raise DatabaseError("Invalid connector for timedelta: %s." % connector)
        return super().combine_duration_expression(connector, sub_expressions)

    def integer_field_range(self, internal_type):
        if internal_type in [
            "PositiveBigIntegerField",
            "PositiveIntegerField",
            "PositiveSmallIntegerField",
        ]:
            return (0, 9223372036854775807)
        return (-9223372036854775808, 9223372036854775807)

    def insert_statement(self, on_conflict=None):
        if on_conflict == OnConflict.IGNORE:
            return "INSERT OR IGNORE INTO"
        return super().insert_statement(on_conflict=on_conflict)

    def on_conflict_suffix_sql(self, fields, on_conflict, update_fields, unique_fields):
        if (
            on_conflict == OnConflict.UPDATE
            and self.connection.features.supports_update_conflicts_with_target
        ):
            return "ON CONFLICT(%s) DO UPDATE SET %s" % (
                ", ".join(map(self.quote_name, unique_fields)),
                ", ".join(
                    [
                        f"{field} = EXCLUDED.{field}"
                        for field in map(self.quote_name, update_fields)
                    ]
                ),
            )
        return super().on_conflict_suffix_sql(
            fields,
            on_conflict,
            update_fields,
            unique_fields,
        )
