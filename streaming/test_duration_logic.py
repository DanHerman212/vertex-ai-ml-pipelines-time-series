import unittest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
from streaming.transform import CalculateTripDuration

class TestDuration(unittest.TestCase):
    def test_duration_calculation(self):
        # Inputs: (trip_id, update_dict)
        inputs = [
            ("T1", {'route_id': 'E', 'stop_id': 'G05S', 'timestamp': 1000, 'trip_id': 'T1'}),
            ("T1", {'route_id': 'E', 'stop_id': 'F11S', 'timestamp': 1600, 'trip_id': 'T1'}),
            ("T2", {'route_id': 'E', 'stop_id': 'F11S', 'timestamp': 2000, 'trip_id': 'T2'}), # No origin
        ]
        
        expected_output = [
            ("E_F11S", {'timestamp': 1600, 'duration': 10.0})
        ]
        
        with TestPipeline() as p:
            output = (p
                | beam.Create(inputs)
                | beam.ParDo(CalculateTripDuration(origin_stop_id="G05S", target_stop_id="F11S"))
            )
            
            assert_that(output, equal_to(expected_output))

if __name__ == '__main__':
    unittest.main()
