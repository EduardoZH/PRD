from fastapi import FastAPI, Response, status, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import List

from model import predict

import re

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

    @field_validator('name')
    @classmethod
    def clean_name(cls, v):
        if v is None:
            return ""
        v = str(v)
        v = re.sub(r'[^a-zA-Z0-9\s\-.,!?]', ' ', v)
        v = re.sub(r'\s+', ' ', v).strip()

        if len(v) > 1200:
            v = v[:1200]
        return v

class Transaction(BaseModel):
    transaction_id: str
    terminal_name: str
    terminal_description: str
    city: str
    amount: float
    items: List[Item]

    @field_validator('terminal_name', 'terminal_description')
    @classmethod
    def clean_text(cls, v):
        if v is None:
            return ""
        v = str(v)
        v = re.sub(r'[^a-zA-Z0-9\s\-.,!?]', ' ', v)
        v = re.sub(r'\s+', ' ', v).strip()
        if len(v) > 1200:
            v = v[:1200]
        return v

    @field_validator('amount')
    @classmethod
    def check_amount(cls, v):
        if v < 0:
            raise ValueError('amount must be non-negative')
        return v
class Transactions(BaseModel):
    transactions: List[Transaction ]

class ModelInfo(BaseModel):
    model_name: str
    model_version: str

@app.get('/health')
async def health_check():
    if True:
        return {'status':'ok'}
    
@app.get('/model/info')
async def model_information():
    return {
        'model_name':'mcc-classifier',
        'model_version':'1.0.0'
    }

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    missing_elements = []
    print(exc.errors())
    for error in exc.errors():
        missing_elements.append(error['loc'][1])
    if all([isinstance(x, str) for x in missing_elements]):
        error_text = 'Missing or invalid ' + ', '.join(missing_elements)
    else:
        error_text = 'Unexpected error: some fields are missing or invalid'

    return JSONResponse(status_code=400, content={'error':error_text})


@app.post('/predict')
async def transaction_classification(response: Response, transaction: Transaction):
    transaction = transaction.model_dump()
    print(transaction)
    prediction = predict(transaction)
    if prediction:
        return prediction
    else:
        raise HTTPException(status_code=503, headers={'Error':'Model not ready'})


@app.post('/predict/batch', status_code=200)
async def batch_classification(response: Response, transactions: Transactions):
    transactions = transactions.model_dump()

    predictions = predict(transactions)
    if predictions:
        return predictions
    else:
        raise HTTPException(status_code=503, headers={'Error':'Model not ready'})