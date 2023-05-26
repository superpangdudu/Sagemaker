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
REGION = 'us-east-2'

INSTANCE_TYPE = 'ml.g4dn.xlarge'
PYTHON_VERSION = 'py38'

MIN_ENDPOINT_CAPACITY = 0
MAX_ENDPOINT_CAPACITY = 10

ENDPOINT = 'WATER-COLOR-V3-3a2b3e96-b94a-4855-8ff6-57e6cfa98251'

boto3_session = boto3.session.Session(
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=ACCESS_KEY_SECRET,
    region_name=REGION
)

cloudwatch_client = boto3_session.client('cloudwatch')
autoscaling_client = boto3_session.client('application-autoscaling')


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

    # response = autoscaling_client.put_scaling_policy(
    #     PolicyName="HasBacklogWithoutCapacity-ScalingPolicy",
    #     ServiceNamespace="sagemaker",
    #     ResourceId=resource_id,
    #     ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    #     PolicyType="StepScaling",
    #     StepScalingPolicyConfiguration={
    #         "AdjustmentType": "ChangeInCapacity",
    #         "MetricAggregationType": "Average",
    #         "Cooldown": 300,
    #         "StepAdjustments": [
    #             {
    #                 "MetricIntervalLowerBound": 0,
    #                 "ScalingAdjustment": 1
    #             }
    #         ]
    #     },
    # )
    #
    # response = cloudwatch_client.put_metric_alarm(
    #     AlarmName='HasBacklogWithoutCapacity',
    #     MetricName='HasBacklogWithoutCapacity',
    #     ActionsEnabled=True,
    #     Namespace='AWS/SageMaker',
    #     Statistic='Average',
    #     EvaluationPeriods=2,
    #     DatapointsToAlarm=2,
    #     Threshold=1,
    #     ComparisonOperator='GreaterThanOrEqualToThreshold',
    #     TreatMissingData='missing',
    #     Dimensions=[
    #         {'Name': 'EndpointName', 'Value': endpoint_name},
    #     ],
    #     Period=60
    # )


make_endpoint_scalable(ENDPOINT, MIN_ENDPOINT_CAPACITY, MAX_ENDPOINT_CAPACITY)