import numpy as np


class BasicApproach:
    @classmethod
    def moving_average_forecast(cls, series, window_size):

        mov = np.cumsum(series)
        mov[window_size:] = mov[window_size:] - mov[:-window_size]
        return mov[window_size - 1:-1] / window_size

    """Forecasts the mean of the last few values.
             If window_size=1, then this is equivalent to naive forecast
             This implementation is *much* faster than the previous one

             # Exemple
      split_time = 240
      created_at_valid = created_at_column_array[:split_time]
      value_valid = value_column_array[:split_time]
      # Moving average forecast
      # moving_avg = BasicApproach.moving_average_forecast(value_column_array, 24)[:split_time]
      # print(Metrics.calculate_mae(value_valid, moving_avg))

             """
    @classmethod
    def moving_average_forecast1(cls, series, window_size):
        """Forecasts the mean of the last few values.
           If window_size=1, then this is equivalent to naive forecast"""
        forecast = []
        for time in range(len(series) - window_size):
            forecast.append(series[time:time + window_size].mean())
        return np.array(forecast)

