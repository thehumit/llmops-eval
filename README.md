### WebUI Test:
```
llmops_evaluation % python generate_samples.py \
     --loader tools/multiturn_dataset_WEBUI.py \
     --endpoint http://10.146.100.205:30001/v1 \
     --api-key your_key_here \
     --output generated.json
```