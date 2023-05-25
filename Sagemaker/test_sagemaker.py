
import boto3
import sagemaker

import uuid
import time
import threading

from sagemaker.pytorch.model import PyTorchModel
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

#########################################################################################
ACCESS_KEY_ID = 'AKIAUHC5UZEKGAES5SEE'
ACCESS_KEY_SECRET = '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'

INSTANCE_TYPE = 'ml.g4dn.xlarge'
PYTHON_VERSION = 'py38'

MIN_ENDPOINT_CAPACITY = 1
MAX_ENDPOINT_CAPACITY = 8

models = [
    ('WATER-COLOR-V3', './code')
]

#########################################################################################
# sts_client = boto3.client(
#     'sts',
#     aws_access_key_id=ACCESS_KEY_ID,
#     aws_secret_access_key=ACCESS_KEY_SECRET
# )
# account_id = sts_client.get_caller_identity().get('Account')

autoscaling_client = boto3.client(
    "application-autoscaling",
    aws_access_key_id='AKIAUHC5UZEKGAES5SEE',
    aws_secret_access_key='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',
    region_name='us-east-2'
)

boto3_session = boto3.session.Session(
    aws_access_key_id='AKIAUHC5UZEKGAES5SEE',
    aws_secret_access_key='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',
    region_name='us-east-2'
)

sagemaker_session = sagemaker.Session(
    boto_session=boto3_session
)

s3 = boto3_session.resource('s3')
bucket = sagemaker_session.default_bucket()
print(f'bucket = {bucket}')

# role = sagemaker.get_execution_role(sagemaker_session)
role = 'arn:aws:iam::290106689812:role/KrlyMgr'
print(f'role = {role}')

#########################################################################################
# TODO
model_name = 'sakistriker/AbyssOrangeMix3'
lora_model = 's3://sagemaker-us-east-1-596030579944/fakemonPokMonLORA/fakemonPokMonLORA_v10Beta.safetensors'

framework_version = '1.10'
py_version = 'py38'

model_environment = {
    'SAGEMAKER_MODEL_SERVER_TIMEOUT': '600',
    'SAGEMAKER_MODEL_SERVER_WORKERS': '1',
    'model_name': model_name,
    'lora_model': lora_model,
    's3_bucket': bucket
}

#########################################################################################
model_path = 's3://sagemaker-us-east-2-290106689812/stablediffusion/assets/model.tar.gz'


#########################################################################################
def deploy_model(name, source_dir):
    model = PyTorchModel(
        sagemaker_session=sagemaker_session,
        name=None,
        model_data=model_path,
        entry_point='inference.py',
        source_dir=source_dir,
        role=role,
        framework_version=framework_version,
        py_version=py_version,
        env=model_environment
    )

    #
    endpoint_name = f'{name}-{str(uuid.uuid4())}'
    async_config = AsyncInferenceConfig(output_path='s3://{0}/{1}/asyncinvoke/out/'.format(bucket, 'stablediffusion'))
    instance_count = 1

    #
    start_time = time.time()
    print(f'start to deploy model: {name}')
    async_predictor = model.deploy(
        endpoint_name=endpoint_name,
        instance_type=INSTANCE_TYPE,
        initial_instance_count=instance_count,
        async_inference_config=async_config,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        # wait=False
    )
    end_time = time.time()
    print(f'\nmodel {name} is deployed, used {end_time - start_time}s')

    return async_predictor


def make_endpoint_scalable(endpoint_name, min_capacity, max_capacity):
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    # Configure Autoscaling on asynchronous endpoint down to zero instances
    response = autoscaling_client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity,
    )

    response = autoscaling_client.put_scaling_policy(
        PolicyName=f'Request-ScalingPolicy-{endpoint_name}',
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 2.0,
            "CustomizedMetricSpecification": {
                "MetricName": "ApproximateBacklogSizePerInstance",
                "Namespace": "AWS/SageMaker",
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                "Statistic": "Average",
            },
            "ScaleInCooldown": 600,  # duration until scale in begins (down to zero)
            "ScaleOutCooldown": 300  # duration between scale out attempts
        },
    )


#########################################################################################
predictors = []


def do_model_deploying(name, src):
    predictor = deploy_model(name, src)
    predictors.append(predictor)
    make_endpoint_scalable(predictor.endpoint_name, MIN_ENDPOINT_CAPACITY, MAX_ENDPOINT_CAPACITY)


threads = []
for (n, s) in models:
    thread = threading.Thread(target=do_model_deploying, args=(n, s))
    thread.start()
    threads.append(thread)

for t in threads:
    t.join()


# for p in predictors:
#     p.delete_endpoint()
# async_predictor.delete_endpoint()
