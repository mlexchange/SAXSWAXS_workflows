# Appears not to be necessary
#prefect config set PREFECT_API_URL=$PREFECT_API_URL
#prefect deployment ls
prefect work-pool create reduction-pool --type "process"
prefect work-pool update reduction-pool --concurrency-limit 1
prefect work-pool create fitting-pool --type "process" 
prefect work-pool create gpCam-pool --type "process" 
prefect deploy --all