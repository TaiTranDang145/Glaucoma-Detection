import unittest

import pandas as pd

from src.domain_sampling import make_domain_balanced_sampler


class DomainBalancedSamplerTests(unittest.TestCase):
    def test_each_domain_gets_equal_total_sampling_weight(self):
        dataframe = pd.DataFrame({"domain": ["HYDR"] * 4 + ["DRISHTI_GS"] * 2})

        sampler = make_domain_balanced_sampler(dataframe["domain"])

        weights = sampler.weights.tolist()
        hydr_weight = sum(weights[:4])
        drishti_weight = sum(weights[4:])
        self.assertAlmostEqual(hydr_weight, drishti_weight)
        self.assertEqual(sampler.num_samples, len(dataframe))
        self.assertTrue(sampler.replacement)


if __name__ == "__main__":
    unittest.main()
