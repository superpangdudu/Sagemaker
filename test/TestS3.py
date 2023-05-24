
import s3fs
import boto3

s3 = boto3.resource('s3')
s3 = boto3.client(
    's3',
    aws_access_key_id='AKIAUHC5UZEKBLUVXKFE',
    aws_secret_access_key='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',
)
s3.download_file('sagemaker-us-east-2-290106689812',
                            'model/lora/handPaintedPortrait_v12.safetensors',
                            './handPaintedPortrait_v12.safetensors')

