from dataclasses import dataclass
import datetime
from string import Formatter
from typing import Any, Dict, List, Optional
import pandas


MaxNAxisLabels = 300


def _format_datetime(dt: Optional[datetime.datetime]) -> Optional[str]:
    if dt is pandas.NaT:
        return None
    else:
        return dt.isoformat() + 'Z'


def python_format_to_d3_tick_format(python_format: str) -> str:
    """
    Build a d3-scale tickFormat specification based on Python str.

    >>> python_format_to_d3_tick_format('{:,.2f}')
    ',.2f'
    >>> # d3-scale likes to mess about with precision. Its "r" format does
    >>> # what we want; if we left it blank, we'd see format(30) == '3e+1'.
    >>> python_format_to_d3_tick_format('{:,}')
    ',r'
    """
    # Formatter.parse() returns Iterable[(literal, field_name, format_spec,
    # conversion)]
    specifier = next(Formatter().parse(python_format))[2]
    if not specifier or specifier[-1] not in 'bcdoxXneEfFgGn%':
        specifier += 'r'
    return specifier


class GentleValueError(ValueError):
    """
    A ValueError that should not display in red to the user.

    On first load, we don't want to display an error, even though the user
    hasn't selected what to chart. So we'll display the error in the iframe:
    we'll be gentle with the user.
    """


@dataclass
class XSeries:
    values: pandas.Series
    column: Any
    """RenderColumn with .name, .type and .format."""

    @property
    def tick_format(self) -> Optional[str]:
        """d3-scale tickFormat specification (if X axis is numeric)."""
        if self.column.format is None:
            return None
        else:
            return python_format_to_d3_tick_format(self.column.format)

    @property
    def vega_data_type(self) -> str:
        if self.column.type == 'datetime':
            return 'temporal'
        elif self.column.type == 'number':
            return 'quantitative'
        else:
            return 'ordinal'

    @property
    def json_compatible_values(self) -> pandas.Series:
        """
        Array of str or int or float values for the X axis of the chart.

        In particular: datetime64 values will be converted to str.
        """
        if self.column.type == 'datetime':
            try:
                utc_series = self.values.dt.tz_convert(None).to_series()
            except TypeError:
                utc_series = self.values

            str_series = utc_series.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            str_series = str_series.mask(self.values.isna())  # 'NaT' => np.nan

            return str_series.values
        else:
            return self.values


@dataclass(frozen=True)
class YSeries:
    series: pandas.Series
    name: str
    format: str
    """Python formatting string, like '{:,.2f}'."""

    @property
    def tick_format(self) -> str:
        """d3-scale tickFormat specification."""
        return python_format_to_d3_tick_format(self.format)


@dataclass(frozen=True)
class Chart:
    """Fully-sane parameters. Columns are series."""

    title: str
    x_axis_label: str
    """d3-scale tickFormat specification (if X axis is numeric)."""

    y_axis_label: str
    x_series: XSeries
    y_column: YSeries
    """d3-scale tickFormat specification."""

    def to_vega_data_values(self) -> List[Dict[str, Any]]:
        """
        Build a dict for Vega's .data.values Array.

        Return value is a list of dict records. Each has
        {'X Name': 1.0, 'Y Name': 1.0}
        """
        # Drop null rows that contain null in either series,
        # since x can be either nparray or df, use pandas to drop
        df = pandas.DataFrame({
            'x': self.x_series.json_compatible_values,
            'y': self.y_column.series
        })
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
            'title': self.x_axis_label,
            'format': self.x_series.tick_format,
        }
        if self.x_series.vega_data_type == 'ordinal':
            x_axis.update({
                'labelAngle': 0,
                'labelOverlap': False,
            })

        ret = {
            '$schema': 'https://vega.github.io/schema/vega-lite/v3.json',
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
                    'axis': {
                        'title': self.y_axis_label,
                        'format': self.y_column.tick_format
                    }
                }
            },
        }

        return ret


@dataclass(frozen=True)
class Form:
    """
    Parameter dict specified by the user: valid types, unchecked values.
    """

    title: str
    x_axis_label: str
    y_axis_label: str
    x_column: str
    y_column: str

    def _make_x_series(self, table: pandas.DataFrame,
                       input_columns: Dict[str, Any]) -> XSeries:
        """
        Create an XSeries ready for charting, or raise ValueError.
        """
        if self.x_column not in table.columns:
            raise GentleValueError('Please choose an X-axis column')

        series = table[self.x_column]
        column = input_columns[self.x_column]
        nulls = series.isna().values
        x_values = table[self.x_column]
        safe_x_values = x_values[~nulls]  # so we can min(), len(), etc

        if column.type == 'text' and len(safe_x_values) > MaxNAxisLabels:
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

        return XSeries(x_values, column)

    def make_chart(self, table: pandas.DataFrame,
                   input_columns: Dict[str, Any]) -> Chart:
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
        x_series = self._make_x_series(table, input_columns)
        x_values = x_series.values
        if not self.y_column:
            raise GentleValueError('Please choose a Y-axis column')
        if self.y_column == self.x_column:
            raise ValueError(
                f'Cannot plot Y-axis column "{self.y_column}" '
                'because it is the X-axis column'
            )

        series = table[self.y_column]

        # Find how many Y values can actually be plotted on the X axis. If
        # there aren't going to be any Y values on the chart, raise an
        # error.
        matches = pandas.DataFrame({'X': x_values, 'Y': series}).dropna()
        if not matches['X'].count():
            raise ValueError(
                f'Cannot plot Y-axis column "{self.y_column}" '
                'because it has no values'
            )

        y_column = YSeries(series, self.y_column,
                           input_columns[self.y_column].format)

        title = self.title or 'Scatter Plot'
        x_axis_label = self.x_axis_label or x_series.name
        y_axis_label = self.y_axis_label or y_column.name

        return Chart(title=title, x_axis_label=x_axis_label,
                     y_axis_label=y_axis_label, x_series=x_series,
                     y_column=y_column)


def render(table, params, *, input_columns):
    form = Form(**params)

    try:
        chart = form.make_chart(table, input_columns)
    except GentleValueError as err:
        return (table, '', {'error': str(err)})
    except ValueError as err:
        return (table, str(err), {'error': str(err)})

    json_dict = chart.to_vega()
    return (table, '', json_dict)
