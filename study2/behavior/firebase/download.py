#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from datetime import datetime, timezone, timedelta
from google.cloud.firestore_v1 import DocumentSnapshot

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    cred = credentials.Certificate(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'serviceAccountKey.json'))
    firebase_admin.initialize_app(cred)
    return firestore.client()

def convert_timestamps(data):
    """Convert Firestore Timestamp objects to KST datetime string"""
    if isinstance(data, dict):
        return {key: convert_timestamps(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_timestamps(item) for item in data]
    elif type(data).__name__ == 'DatetimeWithNanoseconds' or hasattr(data, 'timestamp'):
        # Firestore Timestamp object - convert to KST (UTC+9)
        try:
            dt = data if isinstance(data, datetime) else (data.datetime() if hasattr(data, 'datetime') else datetime.fromtimestamp(data.timestamp()))
            # Convert to KST (UTC+9)
            kst = timezone(timedelta(hours=9))
            dt_kst = dt.astimezone(kst)
            return dt_kst.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return data
    else:
        return data

def download_collection_to_excel(db, collection_name, output_filename=None):
    """
    Download all documents from a Firestore collection to Excel

    Args:
        db: Firestore client
        collection_name: Name of the collection to download
        output_filename: Output filename (default: collection_name_timestamp.xlsx)
    """
    print(f"Fetching data from '{collection_name}' collection...")

    docs = db.collection(collection_name).stream()

    data = []
    for doc in docs:
        doc_data = doc.to_dict()
        doc_data = convert_timestamps(doc_data)
        doc_data['document_id'] = doc.id
        data.append(doc_data)

    if not data:
        print(f"No data found in '{collection_name}' collection.")
        return

    print(f"Found {len(data)} documents.")

    df = pd.DataFrame(data)

    if output_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{collection_name}_{timestamp}.xlsx"

    df.to_excel(output_filename, index=False, engine='openpyxl')
    print(f"Data saved to '{output_filename}'")

    return df

def download_multiple_collections(db, collection_names, output_filename=None):
    """
    Download multiple collections to separate sheets in one Excel file

    Args:
        db: Firestore client
        collection_names: List of collection names
        output_filename: Output filename
    """
    if output_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"firestore_data_{timestamp}.xlsx"

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        for collection_name in collection_names:
            print(f"\nProcessing '{collection_name}' collection...")
            docs = db.collection(collection_name).stream()

            data = []
            for doc in docs:
                doc_data = doc.to_dict()
                doc_data = convert_timestamps(doc_data)
                doc_data['document_id'] = doc.id
                data.append(doc_data)

            if data:
                df = pd.DataFrame(data)
                sheet_name = collection_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  {len(data)} documents saved to '{sheet_name}' sheet.")
            else:
                print(f"  No data in '{collection_name}' collection.")

    print(f"\nAll data saved to '{output_filename}'")

if __name__ == "__main__":
    try:
        # Initialize Firebase
        db = initialize_firebase()

        # Option 1: Download single collection
        download_collection_to_excel(db, 'GameResults')

        # Option 2: Download multiple collections
        # collection_names = ['GameResults', 'Users', 'OtherCollection']
        # download_multiple_collections(db, collection_names, 'all_data.xlsx')

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: You need a Firebase service account key file.")
        print("Generate it from: Firebase Console > Project Settings > Service Accounts")
