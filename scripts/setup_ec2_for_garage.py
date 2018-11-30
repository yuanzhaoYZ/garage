import json
import os
import sys

import boto3
import botocore

from garage import config
from garage.misc import console



def setup_iam():
    iam_client = boto3.client(
        "iam",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=ACCESS_SECRET,
    )
    iam = boto3.resource(
        'iam',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=ACCESS_SECRET)

    # delete existing role if it exists
    try:
        existing_role = iam.Role('garage')
        existing_role.load()
        # if role exists, delete and recreate
        if not query_yes_no(
            ("There is an existing role named garage. "
             "Proceed to delete everything garage-related and recreate?"),
                default="no"):
            sys.exit()
        print("Listing instance profiles...")
        inst_profiles = existing_role.instance_profiles.all()
        for prof in inst_profiles:
            for role in prof.roles:
                print("Removing role %s from instance profile %s" %
                      (role.name, prof.name))
                prof.remove_role(RoleName=role.name)
            print("Deleting instance profile %s" % prof.name)
            prof.delete()
        for policy in existing_role.policies.all():
            print("Deleting inline policy %s" % policy.name)
            policy.delete()
        for policy in existing_role.attached_policies.all():
            print("Detaching policy %s" % policy.arn)
            existing_role.detach_policy(PolicyArn=policy.arn)
        print("Deleting role")
        existing_role.delete()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            pass
        else:
            raise e

    print("Creating role garage")
    iam_client.create_role(
        Path='/',
        RoleName='garage',
        AssumeRolePolicyDocument=json.dumps({
            'Version':
            '2012-10-17',
            'Statement': [{
                'Action': 'sts:AssumeRole',
                'Effect': 'Allow',
                'Principal': {
                    'Service': 'ec2.amazonaws.com'
                }
            }]
        }))

    role = iam.Role('garage')
    print("Attaching policies")
    role.attach_policy(PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess')
    role.attach_policy(
        PolicyArn='arn:aws:iam::aws:policy/ResourceGroupsandTagEditorFullAccess'
    )

    print("Creating inline policies")
    iam_client.put_role_policy(
        RoleName=role.name,
        PolicyName='CreateTags',
        PolicyDocument=json.dumps({
            "Version":
            "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": ["ec2:CreateTags"],
                "Resource": ["*"]
            }]
        }))
    iam_client.put_role_policy(
        RoleName=role.name,
        PolicyName='TerminateInstances',
        PolicyDocument=json.dumps({
            "Version":
            "2012-10-17",
            "Statement": [{
                "Sid": "Stmt1458019101000",
                "Effect": "Allow",
                "Action": ["ec2:TerminateInstances"],
                "Resource": ["*"]
            }]
        }))

    print("Creating instance profile garage")
    iam_client.create_instance_profile(InstanceProfileName='garage', Path='/')
    print("Adding role garage to instance profile garage")
    iam_client.add_role_to_instance_profile(
        InstanceProfileName='garage', RoleName='garage')


def setup_s3():
    from garage.config_personal import S3_BUCKET_NAME
    print("Creating S3 bucket at s3://%s" % S3_BUCKET_NAME)
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=ACCESS_SECRET,
    )
    try:
        s3_client.create_bucket(
            ACL='private',
            Bucket=S3_BUCKET_NAME,
        )
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyExists':
            raise ValueError(
                "Bucket %s already exists. Please reconfigure S3_BUCKET_NAME" %
                S3_BUCKET_NAME) from e
        elif e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print("Bucket already created by you")
        else:
            raise e
    print("S3 bucket created")


def setup_ec2():
    for region in ["us-east-1", "us-west-1", "us-west-2"]:
        print("Setting up region %s" % region)

        ec2 = boto3.resource(
            "ec2",
            region_name=region,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=ACCESS_SECRET,
        )
        ec2_client = boto3.client(
            "ec2",
            region_name=region,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=ACCESS_SECRET,
        )
        existing_vpcs = list(ec2.vpcs.all())
        assert len(existing_vpcs) >= 1
        vpc = existing_vpcs[0]
        print("Creating security group in VPC %s" % str(vpc.id))
        try:
            security_group = vpc.create_security_group(
                GroupName='garage-sg', Description='Security group for garage')
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'InvalidGroup.Duplicate':
                sgs = list(
                    vpc.security_groups.filter(GroupNames=['garage-sg']))
                security_group = sgs[0]
            else:
                raise e

        ALL_REGION_AWS_SECURITY_GROUP_IDS[region] = [security_group.id]

        ec2_client.create_tags(
            Resources=[security_group.id],
            Tags=[{
                'Key': 'Name',
                'Value': 'garage-sg'
            }])
        try:
            security_group.authorize_ingress(
                FromPort=22, ToPort=22, IpProtocol='tcp', CidrIp='0.0.0.0/0')
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'InvalidPermission.Duplicate':
                pass
            else:
                raise e
        print("Security group created with id %s" % str(security_group.id))

        key_name = 'garage-%s' % region
        try:
            print("Trying to create key pair with name %s" % key_name)
            key_pair = ec2_client.create_key_pair(KeyName=key_name)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'InvalidKeyPair.Duplicate':
                if not query_yes_no(
                    ("Key pair with name %s exists. "
                     "Proceed to delete and recreate?") % key_name, "no"):
                    sys.exit()
                print("Deleting existing key pair with name %s" % key_name)
                ec2_client.delete_key_pair(KeyName=key_name)
                print("Recreating key pair with name %s" % key_name)
                key_pair = ec2_client.create_key_pair(KeyName=key_name)
            else:
                raise e

        key_pair_folder_path = os.path.join(config.PROJECT_PATH, "private",
                                            "key_pairs")
        file_name = os.path.join(key_pair_folder_path, "%s.pem" % key_name)

        print("Saving keypair file")
        console.mkdir_p(key_pair_folder_path)
        with os.fdopen(
                os.open(file_name, os.O_WRONLY | os.O_CREAT, 0o600),
                'w') as handle:
            handle.write(key_pair['KeyMaterial'] + '\n')

        # adding pem file to ssh
        os.system("ssh-add %s" % file_name)

        ALL_REGION_AWS_KEY_NAMES[region] = key_name



def setup():
    setup_s3()
    setup_iam()
    setup_ec2()


if __name__ == "__main__":
    setup()
