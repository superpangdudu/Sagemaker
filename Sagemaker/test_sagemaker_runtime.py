
import boto3
import sagemaker

import json

#########################################################################################
ACCESS_KEY_ID = 'AKIAUHC5UZEKGAES5SEE'
ACCESS_KEY_SECRET = '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'
REGION = 'us-east-2'


ENDPOINT = 'WATER-COLOR-V3-3a2b3e96-b94a-4855-8ff6-57e6cfa98251'

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

input_s3 = 's3://sagemaker-us-east-2-290106689812/stablediffusion/asyncinvoke/input/454bd043-bf2c-4584-b6fd-6121f70b5455.json'

sagemaker_client = boto3_session.client('sagemaker-runtime')

for i in range(10):
    response = sagemaker_client.invoke_endpoint_async(
        EndpointName=ENDPOINT,
        InputLocation=input_s3
    )

    print(response)



