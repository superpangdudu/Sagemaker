
import json

event = """
{
	"body": {
		"input_s3_path": "s3://sagemaker-us-east-2-290106689812/stablediffusion/asyncinvoke/input/454bd043-bf2c-4584-b6fd-6121f70b5455.json"
	}
}
"""

event = json.loads(event)
body = event.get("body", "")
print(body)

x = {}
if isinstance(x, dict):
    print('Y')