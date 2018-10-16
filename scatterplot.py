import datetime
import json
from typing import Any, Dict, List, Optional
import pandas
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype


MaxNAxisLabels = 300


def _is_text(series):
    return hasattr(series, 'cat') or series.dtype == object


def _format_datetime(dt: Optional[datetime.datetime]) -> Optional[str]:
    if dt is pandas.NaT:
        return None
    else:
        return dt.isoformat() + 'Z'


class GentleValueError(ValueError):
    """
    A ValueError that should not display in red to the user.

    On first load, we don't want to display an error, even though the user
    hasn't selected what to chart. So we'll display the error in the iframe:
    we'll be gentle with the user.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class XSeries:
    def __init__(self, values: pandas.Series, name: str):
        self.values = values
        self.name = name

    @property
    def vega_data_type(self) -> str:
        if is_datetime64_dtype(self.values.dtype):
            return 'temporal'
        elif is_numeric_dtype(self.values.dtype):
            return 'quantitative'
        else:
            return 'ordinal'

    @property
    def json_compatible_values(self) -> pandas.Series:
        """
        Array of str or int or float values for the X axis of the chart.

        In particular: datetime64 values will be converted to str.
        """
        if is_datetime64_dtype(self.values.dtype):
            return self.values \
                    .astype(datetime.datetime) \
                    .apply(_format_datetime) \
                    .values
        else:
            return self.values


class YSeries:
    def __init__(self, series: pandas.Series, name: str):
        self.series = series
        self.name = name

class Chart:
    """Fully-sane parameters. Columns are series."""
    def __init__(self, *, title: str, x_axis_label: str, y_axis_label: str,
                 x_series: XSeries, y_column: YSeries):
        self.title = title
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.x_series = x_series
        self.y_column = y_column

    def to_vega_data_values(self) -> List[Dict[str, Any]]:
        """
        Build a dict for Vega's .data.values Array.

        Return value is a list of dict records. Each has
        {'X Name': 1.0, 'Y Name': 1.0}
        """
        # Drop null rows that contain null in either series,
        # since x can be either nparray or df, use pandas to drop
        df = pandas.DataFrame({'x': self.x_series.json_compatible_values, 'y': self.y_column.series})
        df.dropna(inplace=True)

        # convert to json acceptable types
        x = df['x'].tolist()
        y = df['y'].tolist()

        data = []
        for idx, value in enumerate(x):
            data.append({'x': value, 'y': y[idx]})

        return data

    def to_vega(self) -> Dict[str, Any]:
        """
        Build a Vega scatter plot.
        """
        x_axis = {
            'title': self.x_axis_label
        }
        if self.x_series.vega_data_type == 'ordinal':
            x_axis.update({
                'labelAngle': 0,
                'labelOverlap': False,
            })

        ret = {
            '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
            'title': self.title,
            'config': {
                'title': {
                    'offset': 15,
                    'color': '#383838',
                    'font': 'Nunito Sans, Helvetica, sans-serif',
                    'fontSize': 20,
                    'fontWeight': 'normal',
                },

                'axis': {
                    'tickSize': 3,
                    'titlePadding': 20,
                    'titleFontSize': 15,
                    'titleFontWeight': 100,
                    'titleColor': '#686768',
                    'titleFont': 'Nunito Sans, Helvetica, sans-serif',
                    'labelFont': 'Nunito Sans, Helvetica, sans-serif',
                    'labelFontWeight': 400,
                    'labelColor': '#383838',
                    'labelFontSize': 12,
                    'labelPadding': 10,
                    'gridOpacity': .5,
                },
            },

            'data': {
                'values': self.to_vega_data_values(),
            },

            'mark': {
                'type': 'point',
                'point': {
                    'shape': 'circle',
                }
            },

            'encoding': {
                'x': {
                    'field': 'x',
                    'type': self.x_series.vega_data_type,
                    'axis': x_axis,
                },

                'y': {
                    'field': 'y',
                    'type': 'quantitative',
                    'axis': {'title': self.y_axis_label}
                }
            },
        }

        return ret


class Form:
    """
    Parameter dict specified by the user: valid types, unchecked values.
    """
    def __init__(self, *, title: str, x_axis_label: str, y_axis_label: str,
                 x_column: str, y_column: str):
        self.title = title
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.x_column = x_column
        self.y_column = y_column

    @staticmethod
    def from_dict(params: Dict[str, Any]) -> 'Form':
        title = str(params.get('title', ''))
        x_axis_label = str(params.get('x_axis_label', ''))
        y_axis_label = str(params.get('y_axis_label', ''))
        x_column = str(params.get('x_column', ''))
        y_column = str(params.get('y_column', ''))
        return Form(title=title, x_axis_label=x_axis_label,
                    y_axis_label=y_axis_label, x_column=x_column,
                    y_column=y_column)

    def _make_x_series(self, table: pandas.DataFrame) -> XSeries:
        """
        Create an XSeries ready for charting, or raise ValueError.
        """
        if self.x_column not in table.columns:
            raise GentleValueError('Please choose an X-axis column')

        series = table[self.x_column]
        nulls = series.isna().values
        x_values = table[self.x_column]
        safe_x_values = x_values[~nulls]  # so we can min(), len(), etc

        if _is_text(x_values) and len(safe_x_values) > MaxNAxisLabels:
            raise ValueError(
                f'Column "{self.x_column}" has {len(safe_x_values)} '
                'text values. We cannot fit them all on the X axis. '
                'Please change the input table to have 10 or fewer rows, or '
                f'convert "{self.x_column}" to number or date.'
            )

        if not len(safe_x_values):
            raise ValueError(
                f'Column "{self.x_column}" has no values. '
                'Please select a column with data.'
            )

        if not len(safe_x_values[safe_x_values != safe_x_values[0]]):
            raise ValueError(
                f'Column "{self.x_column}" has only 1 value. '
                'Please select a column with 2 or more values.'
            )

        x_series = XSeries(x_values, self.x_column)
        return x_series

    def make_chart(self, table: pandas.DataFrame) -> Chart:
        """
        Create a Chart ready for charting, or raise ValueError.

        Features:
        * Error if X column is missing
        * Error if X column does not have two values
        * Error if X column is all-NaN
        * Error if too many X values in text mode (since we can't chart them)
        * X column can be number or date
        * Missing X dates lead to missing records
        * Missing X floats lead to missing records
        * Missing Y values are omitted
        * Error if no Y columns chosen
        * Error if a Y column is missing
        * Error if a Y column is the X column
        * Error if a Y column has fewer than 1 non-missing value
        * Default title, X and Y axis labels
        """
        x_series = self._make_x_series(table)
        x_values = x_series.values
        if not self.y_column:
            raise GentleValueError('Please choose a Y-axis column')

        if self.y_column not in table.columns:
            raise ValueError(
                f'Cannot plot Y-axis column "{self.y_column}" '
                'because it does not exist'
            )
        elif self.y_column == self.x_column:
            raise ValueError(
                f'Cannot plot Y-axis column "{self.y_column}" '
                'because it is the X-axis column'
            )

        series = table[self.y_column]

        if not is_numeric_dtype(series.dtype):
            raise ValueError(
                f'Cannot plot Y-axis column "{self.y_column}" '
                'because it is not numeric. '
                'Convert it to a number before plotting it.'
            )

        # Find how many Y values can actually be plotted on the X axis. If
        # there aren't going to be any Y values on the chart, raise an
        # error.
        matches = pandas.DataFrame({'X': x_values, 'Y': series}).dropna()
        if not matches['X'].count():
            raise ValueError(
                f'Cannot plot Y-axis column "{self.y_column}" '
                'because it has no values'
            )

        y_column = YSeries(series, self.y_column)

        title = self.title or 'Scatter Plot'
        x_axis_label = self.x_axis_label or x_series.name
        y_axis_label = self.y_axis_label or y_column.name

        return Chart(title=title, x_axis_label=x_axis_label,
                     y_axis_label=y_axis_label, x_series=x_series,
                     y_column=y_column)


def render(table, params):
    form = Form.from_dict(params)
    try:
        chart = form.make_chart(table)
    except GentleValueError as err:
        return (table, '', {'error': str(err)})
    except ValueError as err:
        return (table, str(err), {'error': str(err)})

    json_dict = chart.to_vega()
    return (table, '', json_dict)
