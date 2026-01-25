from pickle import load
import pandas as pd

def extract_features_single(data):
    terminal_name = ''.join([x for x in data['terminal_name'] if not x.isdigit()])

    data['LOCATION'] = float(terminal_name=='LOCATION')
    data['PLACE'] = float(terminal_name=='PLACE')
    data['SHOP'] = float(terminal_name=='SHOP')
    data['STORE'] = float(terminal_name=='STORE')
        
    if data['items']:
        data['item_name'] = ' '.join([i['name'] for i in data['items']])
        data['avg_price'] = sum([x['price'] for x in data['items']]) / len(data['items'])
        data['min_price'] = min([x['price'] for x in data['items']])
        data['max_price'] = max([x['price'] for x in data['items']])
        data['item_count'] = len(data['items'])
    else:
        data['item_name'] = ' '
        data['avg_price'] = 0
        data['min_price'] = 0
        data['max_price'] = 0
        data['item_count'] = 0
    transaction = data['transaction_id']

    del data['city']
    del data['transaction_id']
    del data['terminal_name']
    del data['items']

    feature_order = ['item_count', 'LOCATION', 'PLACE', 'SHOP', 'STORE', 'avg_price', 'item_name', 'terminal_description', 'min_price', 'max_price', 'amount']
    df = pd.DataFrame([data])
    df = df[feature_order]

    return (df, transaction)

def exract_batch(batch_data):
    rows = []
    transactions = []
    for data in batch_data['transactions']:
        terminal_name = ''.join([x for x in data['terminal_name'] if not x.isdigit()])

        data['LOCATION'] = float(terminal_name=='LOCATION')
        data['PLACE'] = float(terminal_name=='PLACE')
        data['SHOP'] = float(terminal_name=='SHOP')
        data['STORE'] = float(terminal_name=='STORE')
            
        if data['items']:
            data['item_name'] = ' '.join([i['name'] for i in data['items']])
            data['avg_price'] = sum([x['price'] for x in data['items']]) / len(data['items'])
            data['min_price'] = min([x['price'] for x in data['items']])
            data['max_price'] = max([x['price'] for x in data['items']])
            data['item_count'] = len(data['items'])
        else:
            data['item_name'] = ' '
            data['avg_price'] = 0
            data['min_price'] = 0
            data['max_price'] = 0
            data['item_count'] = 0

        transaction = data['transaction_id']

        del data['city']
        del data['transaction_id']
        del data['terminal_name']
        del data['items']

        rows.append(data)
        transactions.append(transaction)
    feature_order = ['item_count', 'LOCATION', 'PLACE', 'SHOP', 'STORE', 'avg_price', 'item_name', 'terminal_description', 'min_price', 'max_price', 'amount']
    df = pd.DataFrame(rows)
    df = df[feature_order]

    return (df, transactions)

def predict(data):
    with open('mcc-classifier.pkl', 'rb') as file:
        clf = load(file)

    if 'transactions' not in data:
        df, transaction = extract_features_single(data)
        prediction = clf.predict(df)
        confidence = clf.predict_proba(df)
        return {'transaction_id':transaction, 'predicted_mcc':int(prediction[0]), 'confidence':float(f'{confidence[0].max()}'[:4])}
    else:
        df, transactions = exract_batch(data)
        prediction = clf.predict(df)
        confidences = clf.predict_proba(df)
        result = []
        for i, n in enumerate(confidences):
            result.append({'transaction_id':transactions[i], 'predicted_mcc':int(prediction[i]), 'confidence':float(f'{n.max()}'[:4])})
        return {'predictions':result}
    