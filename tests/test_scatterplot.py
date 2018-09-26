#!/usr/bin/env python3

import datetime
import json
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scatterplot import render, Form, GentleValueError


# Minimum valid table
min_table = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, dtype=np.number)


class ConfigTest(unittest.TestCase):
    def assertResult(self, result, expected):
        assert_frame_equal(result[0], expected[0])
        self.assertEqual(result[1], expected[1])
        self.assertEqual(result[2], expected[2])

    def build_form(self, **kwargs):
        params = {
            'title': 'TITLE',
            'x_axis_label': 'X LABEL',
            'y_axis_label': 'Y LABEL',
            'x_column': 'A',
            'y_column': 'B',
        }
        params.update(kwargs)
        return Form(**params)

    def test_missing_x_param(self):
        form = self.build_form(x_column='')
        table = pd.DataFrame({'A': [1, 2], 'B': [2, 3]})
        with self.assertRaisesRegex(
            GentleValueError,
            'Please choose an X-axis column'
        ):
            form.make_chart(table)

    def test_only_one_x_value(self):
        form = self.build_form(x_column='A')
        table = pd.DataFrame({'A': [1, 1], 'B': [2, 3]})
        with self.assertRaisesRegex(
            ValueError,
            'Column "A" has only 1 value. '
            'Please select a column with 2 or more values.'
        ):
            form.make_chart(table)

    def test_no_x_values(self):
        form = self.build_form(x_column='A')
        table = pd.DataFrame({'A': [np.nan, np.nan], 'B': [2, 3]},
                             dtype=np.number)
        with self.assertRaisesRegex(
            ValueError,
            'Column "A" has no values. '
            'Please select a column with data.'
        ):
            form.make_chart(table)

    def test_x_numeric(self):
        form = self.build_form(x_column='A')
        chart = form.make_chart(min_table)
        assert np.array_equal(chart.x_series.values, [1, 2])

        vega = chart.to_vega()
        self.assertEqual(vega['encoding']['x']['type'], 'quantitative')
        self.assertEqual(vega['data']['values'], [
            {'X LABEL': 1, 'Y LABEL': 3},
            {'X LABEL': 2, 'Y LABEL': 4},
        ])

    def test_x_numeric_drop_na_x(self):
        form = self.build_form(x_column='A')
        table = pd.DataFrame({'A': [1, np.nan, 3], 'B': [3, 4, 5]},
                             dtype=np.number)
        chart = form.make_chart(table)
        vega = chart.to_vega()
        self.assertEqual(vega['encoding']['x']['type'], 'quantitative')
        self.assertEqual(vega['data']['values'], [
            {'X LABEL': 1, 'Y LABEL': 3},
            {'X LABEL': 3, 'Y LABEL': 5},
        ])

    def test_x_text(self):
        form = self.build_form(x_column='A')
        chart = form.make_chart(pd.DataFrame({'A': ['a', 'b'], 'B': [1, 2]}))
        assert np.array_equal(chart.x_series.values, ['a', 'b'])

        vega = chart.to_vega()
        self.assertEqual(vega['encoding']['x']['type'], 'ordinal')
        self.assertEqual(vega['data']['values'], [
            {'X LABEL': 'a', 'Y LABEL': 1},
            {'X LABEL': 'b', 'Y LABEL': 2},
        ])

    def test_x_text_drop_na_x(self):
        form = self.build_form(x_column='A')
        table = pd.DataFrame({'A': ['a', None, 'c'], 'B': [1, 2, 3]})
        chart = form.make_chart(table)
        assert np.array_equal(chart.x_series.values, ['a', None, 'c'])

        vega = chart.to_vega()
        self.assertEqual(vega['encoding']['x']['type'], 'ordinal')
        self.assertEqual(vega['data']['values'], [
            {'X LABEL': 'a', 'Y LABEL': 1},
            {'X LABEL': 'c', 'Y LABEL': 3},
        ])

    def test_x_text_too_many_values(self):
        form = self.build_form(x_column='A')
        table = pd.DataFrame({'A': ['a'] * 301, 'B': [1] * 301})
        with self.assertRaisesRegex(
            ValueError,
            'Column "A" has 301 text values. We cannot fit them all on '
            'the X axis. Please change the input table to have 10 or fewer '
            'rows, or convert "A" to number or date.'
        ):
            form.make_chart(table)

    def test_x_datetime(self):
        form = self.build_form(x_column='A')
        t1 = datetime.datetime(2018, 8, 29, 13, 39)
        t2 = datetime.datetime(2018, 8, 29, 13, 40)
        table = pd.DataFrame({'A': [t1, t2], 'B': [3, 4]})
        chart = form.make_chart(table)
        assert np.array_equal(
            chart.x_series.values,
            np.array([t1, t2], dtype='datetime64[ms]')
        )

        vega = chart.to_vega()
        self.assertEqual(vega['encoding']['x']['type'], 'temporal')
        self.assertEqual(vega['data']['values'], [
            {'X LABEL': '2018-08-29T13:39:00Z', 'Y LABEL': 3},
            {'X LABEL': '2018-08-29T13:40:00Z', 'Y LABEL': 4},
        ])

    def test_x_datetime_drop_na_x(self):
        form = self.build_form(x_column='A')
        t1 = datetime.datetime(2018, 8, 29, 13, 39)
        t2 = datetime.datetime(2018, 8, 29, 13, 40)
        table = pd.DataFrame({'A': [t1, None, t2], 'B': [3, 4, 5]})
        chart = form.make_chart(table)
        vega = chart.to_vega()
        self.assertEqual(vega['encoding']['x']['type'], 'temporal')
        self.assertEqual(vega['data']['values'], [
            {'X LABEL': '2018-08-29T13:39:00Z', 'Y LABEL': 3},
            {'X LABEL': '2018-08-29T13:40:00Z', 'Y LABEL': 5},
        ])

    def test_drop_missing_y_but_not_x(self):
        form = self.build_form(x_column='A', y_column='B')
        table = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, np.nan, 6]
        })
        chart = form.make_chart(table)
        vega = chart.to_vega()
        self.assertEqual(vega['data']['values'], [
            {'X LABEL': 1, 'Y LABEL': 4.0},
            {'X LABEL': 3, 'Y LABEL': 6.0}
        ])

    def test_missing_y_param(self):
        form = self.build_form(y_column='')
        with self.assertRaisesRegex(
            GentleValueError,
            'Please choose a Y-axis column'
        ):
            form.make_chart(min_table)

    def test_invalid_y_missing_column(self):
        form = self.build_form(y_column='C')
        with self.assertRaisesRegex(
            ValueError,
            'Cannot plot Y-axis column "C" because it does not exist'
        ):
            form.make_chart(min_table)

    def test_invalid_y_same_as_x(self):
        form = self.build_form(y_column='A')
        with self.assertRaisesRegex(
            ValueError,
            'Cannot plot Y-axis column "A" because it is the X-axis column'
        ):
            form.make_chart(min_table)

    def test_invalid_y_missing_values(self):
        form = self.build_form(
            y_column='C'
        )
        table = pd.DataFrame({
            'A': [1, 2, np.nan, np.nan, 5],
            'B': [4, np.nan, 6, 7, 8],
            'C': [np.nan, np.nan, 9, 10, np.nan],
        })
        with self.assertRaisesRegex(
            ValueError,
            'Cannot plot Y-axis column "C" because it has no values'
        ):
            form.make_chart(table)

    def test_invalid_y_not_numeric(self):
        form = self.build_form(y_column='B')
        table = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
        })
        with self.assertRaisesRegex(
            ValueError,
            'Cannot plot Y-axis column "B" because it is not numeric. '
            'Convert it to a number before plotting it.'
        ):
            form.make_chart(table)

    def test_default_title_and_labels(self):
        form = self.build_form(title='', x_axis_label='', y_axis_label='')
        chart = form.make_chart(min_table)
        vega = chart.to_vega()
        self.assertEqual(vega['title'], 'Scatter Plot')
        self.assertEqual(vega['encoding']['x']['field'], 'A')
        self.assertEqual(vega['encoding']['y']['field'], 'B')

    def test_integration_empty_params(self):
        table = pd.DataFrame({'A': [1, 2], 'B': [2, 3]})
        result = render(table, {})
        self.assertResult(result, (
            table,
            '',
            {'error': 'Please choose an X-axis column'}
        ))

    def test_integration(self):
        table = pd.DataFrame({'A': [1, 2], 'B': [2, 3]})
        result = render(table, {
            'title': 'TITLE',
            'x_column': 'A',
            'x_data_type': '0',
            'y_column': 'B',
            'x_axis_label': 'X LABEL',
            'y_axis_label': 'Y LABEL'
        })
        assert_frame_equal(result[0], table)
        self.assertEqual(result[1], '')
        text = json.dumps(result[2])
        # We won't snapshot the chart: that's too brittle. (We change styling
        # more often than we change logic.) But let's make sure all our
        # parameters are in the JSON.
        self.assertIn('"TITLE"', text)
        self.assertIn('"X LABEL"', text)
        self.assertIn('"Y LABEL"', text)
        self.assertRegex(text, '.*:\s*3[,}]')
