
  
import json
import logging
import os
# import joblib
import pytest
from prediction_service.prediction import form_response, api_response
import prediction_service

input_data = {
    "incorrect_range": 
    {"Present_Price":200,
    "Kms_Driven":500000000000,
    "Owner":5,
    "no_year":10,
    "Fuel_Type_Diesel":1,
    "Fuel_Type_Petrol":1,
    "Seller_Type_Individual":1,
    "Transmission_Manual":1},

    "correct_range":
    {"Present_Price":10,
    "Kms_Driven":50000,
    "Owner":2,
    "no_year":10,
    "Fuel_Type_Diesel":1,
    "Fuel_Type_Petrol":1,
    "Seller_Type_Individual":1,
    "Transmission_Manual":1},

    "incorrect_col":
    {"Present Price":10,
    "Kms Driven":50000,
    "Owner":2,
    "no year":10,
    "Fuel Type Diesel":1,
    "Fuel Type Petrol":1,
    "Seller Type Individual":1,
    "Transmission Manual":1}
}

TARGET_range = {
    "min": 1.0,
    "max": 35.0
}

def test_form_response_correct_range(data=input_data["correct_range"]):
    res = form_response(data)
    assert  TARGET_range["min"] <= res <= TARGET_range["max"]

def test_api_response_correct_range(data=input_data["correct_range"]):
    res = api_response(data)
    assert  TARGET_range["min"] <= res["response"] <= TARGET_range["max"]

def test_form_response_incorrect_range(data=input_data["incorrect_range"]):
    with pytest.raises(prediction_service.prediction.NotInRange):
        res = form_response(data)

def test_api_response_incorrect_range(data=input_data["incorrect_range"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInRange().message

def test_api_response_incorrect_col(data=input_data["incorrect_col"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInCols().message