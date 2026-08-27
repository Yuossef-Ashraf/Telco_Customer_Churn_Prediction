"""Unit tests for Telco_Customer_Churn_Prediction."""
import pytest
from unittest.mock import patch, MagicMock
import pipeline


def test_load_dataset_missing():
    with pytest.raises(FileNotFoundError):
        pipeline.load_dataset("non_existent_data_file_99.csv")


def test_cli_parsing():
    with patch("pipeline.train_and_evaluate") as mock_train:
        with patch("sys.argv", ["pipeline.py", "--data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"]):
            pipeline.main()
            assert mock_train.called
