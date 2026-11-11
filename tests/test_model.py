import unittest
import numpy as np
import pandas as pd
from demo import synthetic_history, synthetic_errors, synthetic_fallback
from model import TARGET, CORE, GAS, OUTAGES, REGIONAL_LOAD, prepare_data, fit_model, predict, natural_spline


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = synthetic_history()
        cls.model = fit_model(cls.history, synthetic_errors(cls.history), '2025-01-01', synthetic_fallback(cls.history))
        cls.inputs = cls.history[cls.history.opr_date.eq('2025-01-01')].copy()

    def test_no_target_dependency(self):
        a, b = self.inputs.copy(), self.inputs.copy()
        a[TARGET], b[TARGET] = np.nan, 999999
        np.testing.assert_allclose(predict(self.model, a).prediction, predict(self.model, b).prediction)

    def test_all_routes_and_finite_prices(self):
        cases = [(CORE, 'calendar'), (GAS, 'month_core'), (REGIONAL_LOAD, 'net_demand'),
                 (['region_COMED'], 'dominion')]
        for columns, route in cases:
            with self.subTest(route=route):
                inputs = self.inputs.copy()
                inputs[columns] = np.nan
                result = predict(self.model, inputs)
                self.assertTrue(result.route.eq(route).all())
                self.assertTrue(np.isfinite(result.prediction).all())
                self.assertTrue(result.correction.eq(0).all())
        result = predict(self.model, self.inputs)
        self.assertEqual(result.route.value_counts().to_dict(), {'full_hybrid': 22, 'geographic': 2})
        self.assertTrue(result.loc[result.route.eq('geographic'), 'correction'].eq(0).all())

    def test_fit_dates_before_forecast(self):
        self.assertLess(self.model['fit_max_date'], '2025-01-01')
        self.assertLess(self.model['tree_fit_max_date'], '2025-01-01')

    def test_ramps_do_not_cross_days_or_missing_hours(self):
        inputs = self.inputs.drop(self.inputs.index[10]).copy()
        inputs = prepare_data(inputs)
        self.assertTrue(inputs.loc[inputs.hour_beginning.eq(0), 'ramp_in'].isna().all())
        self.assertTrue(inputs.loc[inputs.hour_beginning.eq(23), 'ramp_ahead'].isna().all())
        self.assertTrue(inputs.loc[inputs.hour_beginning.eq(9), 'ramp_ahead'].isna().all())
        self.assertTrue(inputs.loc[inputs.hour_beginning.eq(11), 'ramp_in'].isna().all())

    def test_natural_spline_linear_tails(self):
        knots = [0, 1, 2, 3, 4, 5, 6]
        for values in ([-3, -2, -1], [7, 8, 9]):
            curves = natural_spline(values, knots)
            np.testing.assert_allclose(np.diff(curves, n=2, axis=0), 0, atol=1e-10)


if __name__ == '__main__': unittest.main()
