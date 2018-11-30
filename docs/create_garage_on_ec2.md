```bash
export AWS_ACCESS_KEY=XXX
export AWS_ACCESS_SECRET=XXX+NpzIIZU7Rplf0nPbZNzHopXOhduEl
export GARAGE_S3_BUCKET=garage_test
export AWS_REGION_NAME=us-east-1
export USE_GPU=True

git clone https://github.com/yuanzhaoYZ/garage.git
cd garage
pip install boto3
pip install -e .
echo "yes" | python scripts/setup_ec2_for_garage.py
echo "yes" | python scripts/setup_ec2_for_garage.py
```
