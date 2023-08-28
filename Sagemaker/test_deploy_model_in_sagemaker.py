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
# the id of the user who has the permission to access Sagemaker
ACCESS_KEY_ID = 'AKIA3SVJGCUCB5HWDUYI'
# the secret key of the user who has the permission to access Sagemaker
ACCESS_KEY_SECRET = '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'
# the region of the Sagemaker
REGION = 'us-east-1'

# the instance type of the Sagemaker endpoint
INSTANCE_TYPE = 'ml.g4dn.2xlarge'
# python version used on the Sagemaker endpoint
PYTHON_VERSION = 'py38'

# the instance count of the Sagemaker endpoint
MIN_ENDPOINT_CAPACITY = 0
MAX_ENDPOINT_CAPACITY = 8

models = [
    #('TEST-WATERCOLOR-V3', './code_watercolor_v3', 'inference-watercolor_v3.py'),
    #('TEST-OILPAINTING-V8', './code_oil-painting_v8', 'inference-oilpainting_v8.py'),
    #('OILPAINTING-V9', './code_oil-painting_v9', 'inference-oilpainting_v9.py'),
    #('WATERCOLOR-V5', './code_watercolor_v5', 'inference-watercolor_v5.py'),
    ('DIGITALOIL-V1', './code_digitaloil_v1', 'inference-digitaloil_v1.py'),
]

#########################################################################################
boto3_session = boto3.session.Session(
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=ACCESS_KEY_SECRET,
    region_name=REGION
)

cloudwatch_client = boto3_session.client('cloudwatch')
autoscaling_client = boto3_session.client('application-autoscaling')

# print(boto3_session.get_available_services())

sagemaker_session = sagemaker.Session(
    boto_session=boto3_session
)

s3 = boto3_session.resource('s3')
bucket = sagemaker_session.default_bucket()
print(f'bucket = {bucket}')

# role = sagemaker.get_execution_role(sagemaker_session)
#role = 'arn:aws:iam::290106689812:role/KrlyMgr'
role = 'arn:aws:iam::795997508868:role/KrlyMgr'

#########################################################################################
# TODO Blow config is useless
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
model_path = f's3://{bucket}/stablediffusion/assets/model.tar.gz'


#########################################################################################
def deploy_model(name, source_dir, entry):
    model = PyTorchModel(
        sagemaker_session=sagemaker_session,
        name=None,
        model_data=model_path,
        entry_point=entry,
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
    print(f'\nmodel {name} is deployed on {endpoint_name}, used {end_time - start_time}s')

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
            "TargetValue": 1.2,
            "CustomizedMetricSpecification": {
                "MetricName": "ApproximateBacklogSizePerInstance",
                "Namespace": "AWS/SageMaker",
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                "Statistic": "Average",
            },
            "ScaleInCooldown": 3600,  # duration until scale in begins (down to zero)
            "ScaleOutCooldown": 1800  # duration between scale out attempts
        },
    )

    response = autoscaling_client.put_scaling_policy(
        PolicyName="HasBacklogWithoutCapacity-ScalingPolicy-" + endpoint_name,
        ServiceNamespace="sagemaker",  # The namespace of the service that provides the resource.
        ResourceId=resource_id,  # Endpoint name
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",  # SageMaker supports only Instance Count
        PolicyType="StepScaling",  # 'StepScaling' or 'TargetTrackingScaling'
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ChangeInCapacity",
            # Specifies whether the ScalingAdjustment value in the StepAdjustment property is an absolute number or a percentage of the current capacity.
            "MetricAggregationType": "Maximum",  # The aggregation type for the CloudWatch metrics.
            "Cooldown": 1800,
            # The amount of time, in seconds, to wait for a previous scaling activity to take effect.
            "StepAdjustments":  # A set of adjustments that enable you to scale based on the size of the alarm breach.
                [
                    {
                        "MetricIntervalLowerBound": 0,
                        "ScalingAdjustment": 1
                    }
                ]
        },
    )
    step_scaling_policy_arn = response["PolicyARN"]
    step_scaling_policy_alarm_name = "ScaleOutFromZero-" + endpoint_name
    response = cloudwatch_client.put_metric_alarm(
        AlarmName=step_scaling_policy_alarm_name,
        MetricName='HasBacklogWithoutCapacity',
        Namespace='AWS/SageMaker',
        Statistic='Average',
        EvaluationPeriods=2,
        DatapointsToAlarm=2,
        Threshold=1,
        ComparisonOperator='GreaterThanOrEqualToThreshold',
        TreatMissingData='missing',
        Dimensions=[
            {'Name': 'EndpointName', 'Value': endpoint_name},
        ],
        Period=60,
        AlarmActions=[step_scaling_policy_arn]
    )


#########################################################################################
def test_predict(predictor):
    from sagemaker.async_inference.waiter_config import WaiterConfig

    # for oilpainting
    # inputs_txt2img = {
    #     "prompt": "oilpainting\(sargent\), masterpiece, oilpainting style of sargent, a painting of ",
    #     "negative_prompt": "nsfw, worst quality, zombie, red skin, hole, bad anatomy, blush",
    #     #"sampler": "dpm2_a",
    #     "seed": 52362,
    #     "height": 1080,
    #     "width": 720,
    #     "count": 2,
    #     "input_image": f's3://{bucket}/image/test.jpg'
    # }
    # for digitaloil
    inputs_txt2img = {
        "prompt": " brush strokes abstract, by John Berkey, ",
        "negative_prompt": "nsfw, worst quality, zombie, red skin, hole, bad anatomy, blush, (greyscale, worst quality),EasyNegativeV2, blue eyes, blurry,",
        # "sampler": "dpm2_a",
        "seed": 52362,
        "height": 1080,
        "width": 720,
        "count": 2,
        "input_image": f's3://{bucket}/image/test.jpg'
    }
    # for watercolor
    # inputs_txt2img = {
    #     "prompt": "watercolor \(style\), a watercolor painting of ",
    #     "negative_prompt": "nsfw, worst quality, zombie, red skin, hole, bad anatomy, blush, (greyscale, worst quality),EasyNegativeV2, blue eyes, blurry,",
    #     # "sampler": "dpm2_a",
    #     "seed": 52362,
    #     "height": 1080,
    #     "width": 720,
    #     "count": 2,
    #     "input_image": f's3://{bucket}/image/test.jpg'
    # }

    response = predictor.predict_async(inputs_txt2img)

    print(f"Response object: {response}")
    print(f"Response output path: {response.output_path}")
    print("Start Polling to get response:")

    start = time.time()
    config = WaiterConfig(
        max_attempts=100,  # number of attempts
        delay=10  # time in seconds to wait between attempts
    )

    result = response.get_result(config)
    print(f'{result}')
    print(f"Time taken: {time.time() - start}s")


#########################################################################################
predictors = []


def do_model_deploying(name, src, entry):
    predictor = deploy_model(name, src, entry)
    predictors.append(predictor)
    make_endpoint_scalable(predictor.endpoint_name, MIN_ENDPOINT_CAPACITY, MAX_ENDPOINT_CAPACITY)
    test_predict(predictor)


threads = []
for (n, s, e) in models:
    thread = threading.Thread(target=do_model_deploying, args=(n, s, e))
    thread.start()
    threads.append(thread)

for t in threads:
    t.join()

# for p in predictors:
#     p.delete_endpoint()
# async_predictor.delete_endpoint()
