
import boto3
import sagemaker

import json

#########################################################################################
ACCESS_KEY_ID = 'AKIAUHC5UZEKGAES5SEE'
ACCESS_KEY_SECRET = '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'
REGION = 'us-east-2'


ENDPOINT = 'OIL-PAINTING-V8-4f3a6968-e7c1-4d9e-ac76-c9fc221a27d4'

# input_values = {
#     "prompt": "a photo of an astronaut riding a horse on mars",
#     "negative_prompt": "",
#     "steps": 20,
#     "sampler": "euler_a",
#     "seed": 52362,
#     "height": 512,
#     "width": 512,
#     "count": 2,
#     "input_image": 's3://sagemaker-us-east-2-290106689812/image/input_image_vermeer.png'
# }
# json_data = json.dumps(input_values)

#########################################################################################
boto3_session = boto3.session.Session(
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=ACCESS_KEY_SECRET,
    region_name=REGION
)

input_s3 = 's3://sagemaker-us-east-2-290106689812/stablediffusion/asyncinvoke/input/005078b5-254e-4a3d-a5d6-073d5027b753.json'

sagemaker_client = boto3_session.client('sagemaker-runtime')

for i in range(10):
    response = sagemaker_client.invoke_endpoint_async(
        EndpointName=ENDPOINT,
        InputLocation=input_s3
    )

    print(response)



