#!/usr/bin/env python3

from collections import namedtuple
import datetime
import json
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from scatterplot import (
    render,
    Form,
    GentleValueError,
)
from cjwmodule.testing.i18n import i18n_message


Column = namedtuple("Column", ("name", "type", "format"))


# Minimum valid table
min_table = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, dtype=np.number)
min_columns = {"A": Column("A", "number", "{:,}"), "B": Column("B", "number", "{:,}")}


class ConfigTest(unittest.TestCase):
    def assertResult(self, result, expected):
        assert_frame_equal(result[0], expected[0])
        self.assertEqual(result[1], expected[1])
        self.assertEqual(result[2], expected[2])

    def build_form(self, **kwargs):
        params = {
            "title": "TITLE",
            "x_axis_label": "X LABEL",
            "y_axis_label": "Y LABEL",
            "x_column": "A",
            "y_column": "B",
        }
        params.update(kwargs)
        return Form(**params)

    def test_missing_x_param(self):
        form = self.build_form(x_column="")
        with self.assertRaises(GentleValueError) as cm:
            form.make_chart(
                pd.DataFrame({"A": [1, 2], "B": [2, 3]}),
                {
                    "A": Column("A", "number", "{:,d}"),
                    "B": Column("B", "number", "{:,d}"),
                },
            )
        self.assertEqual(
            cm.exception.i18n_message, i18n_message("noXAxisError.message")
        )

    def test_only_one_x_value(self):
        form = self.build_form(x_column="A")
        with self.assertRaises(GentleValueError) as cm:
            form.make_chart(
                pd.DataFrame({"A": [1, 1], "B": [2, 3]}),
                {
                    "A": Column("A", "number", "{:,d}"),
                    "B": Column("B", "number", "{:,d}"),
                },
            )
        self.assertEqual(
            cm.exception.i18n_message,
            i18n_message("onlyOneValueError.message", {"column_name": "A"}),
        )

    def test_no_x_values(self):
        form = self.build_form(x_column="A")
        with self.assertRaises(GentleValueError) as cm:
            form.make_chart(
                pd.DataFrame({"A": [np.nan, np.nan], "B": [2, 3]}, dtype=np.float64),
                {
                    "A": Column("A", "number", "{:,d}"),
                    "B": Column("B", "number", "{:,d}"),
                },
            )
        self.assertEqual(
            cm.exception.i18n_message,
            i18n_message("noValuesError.message", {"column_name": "A"}),
        )

    def test_x_numeric(self):
        form = self.build_form(x_column="A")
        chart = form.make_chart(min_table, min_columns)
        assert np.array_equal(chart.x_series.values, [1, 2])

        vega = chart.to_vega()
        self.assertEqual(vega["encoding"]["x"]["type"], "quantitative")
        self.assertEqual(
            vega["data"]["values"],
            [
                {"x": 1, "y": 3},
                {"x": 2, "y": 4},
            ],
        )

    def test_x_numeric_drop_na_x(self):
        form = self.build_form(x_column="A")
        table = pd.DataFrame({"A": [np.nan, 2, 3], "B": [3, 4, 5]}, dtype=np.number)
        chart = form.make_chart(
            table,
            {"A": Column("A", "number", "{:,d}"), "B": Column("B", "number", "{:,d}")},
        )
        vega = chart.to_vega()
        self.assertEqual(vega["encoding"]["x"]["type"], "quantitative")
        self.assertEqual(
            vega["data"]["values"],
            [
                {"x": 2, "y": 4},
                {"x": 3, "y": 5},
            ],
        )

    def test_x_text(self):
        form = self.build_form(x_column="A")
        chart = form.make_chart(
            pd.DataFrame({"A": ["a", "b"], "B": [1, 2]}),
            {"A": Column("A", "text", None), "B": Column("B", "number", "{:,d}")},
        )
        assert np.array_equal(chart.x_series.values, ["a", "b"])

        vega = chart.to_vega()
        self.assertEqual(vega["encoding"]["x"]["type"], "ordinal")
        self.assertEqual(
            vega["data"]["values"],
            [
                {"x": "a", "y": 1},
                {"x": "b", "y": 2},
            ],
        )

    def test_x_text_drop_na_x(self):
        form = self.build_form(x_column="A")
        table = pd.DataFrame({"A": ["a", None, "c"], "B": [1, 2, 3]})
        chart = form.make_chart(
            table, {"A": Column("A", "text", None), "B": Column("B", "number", "{:,d}")}
        )
        assert np.array_equal(chart.x_series.values, ["a", None, "c"])

        vega = chart.to_vega()
        self.assertEqual(vega["encoding"]["x"]["type"], "ordinal")
        self.assertEqual(
            vega["data"]["values"],
            [
                {"x": "a", "y": 1},
                {"x": "c", "y": 3},
            ],
        )

    def test_x_text_too_many_values(self):
        form = self.build_form(x_column="A")
        table = pd.DataFrame({"A": ["a"] * 301, "B": [1] * 301})
        with self.assertRaises(GentleValueError) as cm:
            form.make_chart(
                pd.DataFrame({"A": ["a"] * 301, "B": [1] * 301}),
                {"A": Column("A", "text", None), "B": Column("B", "number", "{:,d}")},
            )
        self.assertEqual(
            cm.exception.i18n_message,
            i18n_message(
                "tooManyTextValuesError.message",
                {
                    "x_column": "A",
                    "n_safe_x_values": 301,
                },
            ),
        )

    def test_x_timestamp(self):
        form = self.build_form(x_column="A")
        t1 = datetime.datetime(2018, 8, 29, 13, 39)
        t2 = datetime.datetime(2018, 8, 29, 13, 40)
        table = pd.DataFrame({"A": [t1, t2], "B": [3, 4]})
        chart = form.make_chart(
            table,
            # TODO support datetime format
            {"A": Column("A", "timestamp", None), "B": Column("B", "number", "{:,d}")},
        )
        assert_series_equal(chart.x_series.values, pd.Series([t1, t2], name="A"))

        vega = chart.to_vega()
        self.assertEqual(vega["encoding"]["x"]["type"], "temporal")
        self.assertEqual(
            vega["data"]["values"],
            [
                {"x": "2018-08-29T13:39:00Z", "y": 3},
                {"x": "2018-08-29T13:40:00Z", "y": 4},
            ],
        )

    def test_x_timestamp_drop_na_x(self):
        form = self.build_form(x_column="A")
        t1 = datetime.datetime(2018, 8, 29, 13, 39)
        t2 = datetime.datetime(2018, 8, 29, 13, 40)
        table = pd.DataFrame({"A": [t1, None, t2], "B": [3, 4, 5]})
        chart = form.make_chart(
            table,
            # TODO support datetime format
            {"A": Column("A", "timestamp", None), "B": Column("B", "number", "{:,d}")},
        )
        vega = chart.to_vega()
        self.assertEqual(vega["encoding"]["x"]["type"], "temporal")
        self.assertEqual(
            vega["data"]["values"],
            [
                {"x": "2018-08-29T13:39:00Z", "y": 3},
                {"x": "2018-08-29T13:40:00Z", "y": 5},
            ],
        )

    def test_drop_missing_y_but_not_x(self):
        form = self.build_form(x_column="A", y_column="B")
        table = pd.DataFrame({"A": [1, 2, 3], "B": [4, np.nan, 6]})
        chart = form.make_chart(
            table,
            {"A": Column("A", "number", "{:,f}"), "B": Column("B", "number", "{:,d}")},
        )
        vega = chart.to_vega()
        self.assertEqual(
            vega["data"]["values"], [{"x": 1, "y": 4.0}, {"x": 3, "y": 6.0}]
        )

    def test_missing_y_param(self):
        form = self.build_form(y_column="")
        with self.assertRaises(GentleValueError) as cm:
            form.make_chart(min_table, min_columns)
        self.assertEqual(
            cm.exception.i18n_message, i18n_message("noYAxisError.message")
        )

    def test_invalid_y_same_as_x(self):
        form = self.build_form(y_column="A")
        with self.assertRaises(GentleValueError) as cm:
            form.make_chart(min_table, min_columns)
        self.assertEqual(
            cm.exception.i18n_message,
            i18n_message("sameAxesError.message", {"column_name": "A"}),
        )

    def test_invalid_y_missing_values(self):
        form = self.build_form(y_column="C")
        table = pd.DataFrame(
            {
                "A": [1, 2, np.nan, np.nan, 5],
                "B": [4, np.nan, 6, 7, 8],
                "C": [np.nan, np.nan, 9, 10, np.nan],
            }
        )
        with self.assertRaises(GentleValueError) as cm:
            form.make_chart(
                table,
                {
                    "A": Column("A", "number", "{:}"),
                    "B": Column("B", "number", "{:}"),
                    "C": Column("C", "number", "{:}"),
                },
            )
        self.assertEqual(
            cm.exception.i18n_message,
            i18n_message("emptyAxisError.message", {"column_name": "C"}),
        )

    def test_default_title_and_labels(self):
        form = self.build_form(title="", x_axis_label="", y_axis_label="")
        chart = form.make_chart(min_table, min_columns)
        vega = chart.to_vega()
        self.assertEqual(vega["title"], "Scatter Plot")
        self.assertEqual(vega["encoding"]["x"]["field"], "x")
        self.assertEqual(vega["encoding"]["y"]["field"], "y")

    def test_integration_empty_params(self):
        table = pd.DataFrame({"A": [1, 2], "B": [2, 3]})
        result = render(
            table,
            {
                "title": "",
                "x_column": "",
                "y_column": "",
                "x_axis_label": "",
                "y_axis_label": "",
            },
            input_columns={
                "A": Column("A", "number", "{:d}"),
                "B": Column("B", "number", "{:d}"),
            },
        )
        self.assertResult(
            result,
            (
                table,
                i18n_message("noXAxisError.message"),
                {"error": "Please correct the error in this step's data or parameters"},
            ),
        )

    def test_integration(self):
        result = render(
            pd.DataFrame({"A": [1, 2], "B": [2, 3]}),
            {
                "title": "TITLE",
                "x_column": "A",
                "y_column": "B",
                "x_axis_label": "X LABEL",
                "y_axis_label": "Y LABEL",
            },
            input_columns={
                "A": Column("A", "number", "{:,.2f}"),
                "B": Column("B", "number", "{:,}"),
            },
        )
        assert_frame_equal(result[0], pd.DataFrame({"A": [1, 2], "B": [2, 3]}))
        self.assertEqual(result[1], "")
        text = json.dumps(result[2])
        # We won't snapshot the chart: that's too brittle. (We change styling
        # more often than we change logic.) But let's make sure all our
        # parameters are in the JSON.
        self.assertIn('"TITLE"', text)
        self.assertIn('"X LABEL"', text)
        self.assertIn('"Y LABEL"', text)
        self.assertRegex(text, ".*:\s*3[,}]")
        self.assertEqual(result[2]["encoding"]["x"]["axis"]["format"], ",.2f")
        self.assertEqual(result[2]["encoding"]["y"]["axis"]["format"], ",r")


if __name__ == "__main__":
    unittest.main()
