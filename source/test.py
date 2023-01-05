import os
import boto3
import wget
import json
import sys
import numpy as np
from cv2 import imread, resize, IMREAD_GRAYSCALE


stack_name = sys.argv[1]
commit_id = sys.argv[2]
endpoint_name = f"{stack_name}-{commit_id[:7]}"

runtime = boto3.client("runtime.sagemaker")

IMAGE_URL = "https://mlops-pipeline-data-488176068240.s3.eu-west-1.amazonaws.com/MNIST_6_0.png?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjELf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCmFwLXNvdXRoLTEiRzBFAiBxZMWSvc0bgy9p9ZXbWEKqljo%2FuQY%2BtQyN0ewj3OMzygIhANomHsX4j%2FhRd2WgWUPANCBTnH3JLIU%2BdR4LAcarc3PWKoUDCBAQAxoMNDg4MTc2MDY4MjQwIgz5jErkbVoycBqyWHsq4gIhHMdHwJz72eX8488WV%2FTG%2FghoW6MJl4xfkiw3wzrfs0A40vHip4UsfQIswU0O5l9gPRT6AQdAbrpA7YYdIt9FzSN6%2FlBK9BGS9aNd74VCwmkx80259WbqlZD%2BU3xIJUEe470XctBfLHnPEUkqpH9BBPD8pADQCJSricZl94RAnXT1EyaLlLWHGYZOX2aAAr25%2BlGqI4YrypseXGlac5jp11U4Gr0kifrPRjlT9%2Fq5mY9%2FB4eb4IcvYu0fftvx9n7OpNC9w%2BTrZqW8qwoc97P7S44Vipj4Hbm7GVb%2FlDGNgZ5kf1uFpBjyMuIGfpL%2FngdXICI%2FWOe52RN5MFvVnm8L5Rxy9hFuX9D%2BbAkp2tdEWfmv4oXfrk3VG8KkbR1N6iXaeIJqU9A9MVh1a%2F8ywNm9wyvlLlRL2nOYZerUXbV0m4yhsu5KAvO%2FKsszRlK9g6Yw49Zds6Mnyjjza6eQtPMVracwpOTZnQY6swJ0cTvuZ82oop5poE1VurqnWbvAkYmUcTpChZhFNKjvGPAR89PUZcbY7SDhb2yWTlBJqdydlYb3wQMfVssygQ4v2z0NnJD7QyXJ0A%2FTZ2eVXFkSf52pifB3JY4k7%2F%2B7Yltew9yyr78Zu%2BL%2Bo7IpcpcmbxWeP%2FO2t8SONno9qnBobG2TgBNU9aa0eGT39YUFr%2BMZ5cfePM8T20J2%2BpqydjAaazoMCvnVTIYnX%2BH9gHXIHIX0W1FsXf2VuJpHy8mpg8P9Vt4cvzcXRDVO8Sygtqu%2FaPPlNQ6KdmEbC2r1aZf05MBw46hIPPPNc5FK1pVnUuONNa3zYlcUxA3AXyPmrhKrchx96GJsL06%2BnkjzJVIANWzSJjPqHjEBPlyIS3%2FAU2Hy1AAcorLdhaP%2B8m4GpnGlufgc&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230105T104517Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAXDKMPU2IGVUAU7MT%2F20230105%2Feu-west-1%2Fs3%2Faws4_request&X-Amz-Signature=5f1f194a77bfadcaaf6b0b46047c5f3af34f1ad288070fb15fb65259ae26e0a7"
test_file = "test.jpg"
wget.download(
    IMAGE_URL,
    test_file,
)

image = imread(test_file, IMREAD_GRAYSCALE)
image = resize(image, (28, 28))
image = image.astype("float32")
image = image.reshape(1, 1, 28, 28)

payload = json.dumps(image.tolist())
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name, Body=payload, ContentType="application/json"
)

result = response["Body"].read()
result = json.loads(result.decode("utf-8"))
print(f"Probabilities: {result}")

np_result = np.asarray(result)
prediction = np_result.argmax(axis=1)[0]
print(f"This is your number: {prediction}")

if prediction != 5:
    print("Model prediction failed.")
    sys.exit(1)

if os.path.exists(test_file):
    os.remove(test_file)
else:
    print("The file does not exist")
