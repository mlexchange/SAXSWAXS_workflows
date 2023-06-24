# Build the deployment definition file flowname-deployment.yaml
#prefect deployment build ./saxswaxs-workflows/flows/file.py:function -n flowname -q workqeuename
# Create deployment
#prefect deployment apply flowname-deployment.yaml