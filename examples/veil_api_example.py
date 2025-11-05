#!/usr/bin/env python3
"""
Python script equivalent to the curl command for making a POST request to the mask endpoint.
"""

import requests
import json
import sys

def make_mask_request(text, entity_cache):
    """
    Make a POST request to the mask endpoint with the given text and entity cache.

    Args:
        text (str): The text to process
        entity_cache (dict): The entity cache containing NAME and COMPANY mappings

    Returns:
        dict: The JSON response from the server
    """
    url = "http://0.0.0.0:8000/mask"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "entity_cache": entity_cache
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}", file=sys.stderr)
            print(f"Response text: {e.response.text}", file=sys.stderr)
        return None

def main():
    # 1st request in conversation
    text = "Louise, a Roche employee with email louise@roche.com, wanted an Apple computer"
    entity_cache = None
    
    result = make_mask_request(text, entity_cache)
    # result.text: "[NAME1], an employee of [COMPANY1], with email [EMAIL1], wanted a [COMPANY2] computer"
    print("Response 1:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # # 2nd request in conversation
    text = "Louise entered the Apple store and saw that they only sold Samsung and Roche products! And she greeted Maria"
    # entity_cache can be programmatically constructed from result.entities
    entity_cache = {
        "NAME": {
            "1": ["Louise"]
        },
        "COMPANY": {
            "1": ["Roche"],
            "2": ["Apple"]
        },
        "EMAIL": {
            "1": ["louise@roche.com"]
        }
    }
    
    result = make_mask_request(text, entity_cache)
    # result.text: "[NAME1] entered the [COMPANY2] store and saw that they only sold [COMPANY3] and [COMPANY1] products! And greeted [NAME2]"
    print("Response 2:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
