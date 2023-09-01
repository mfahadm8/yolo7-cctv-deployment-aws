
mkfile_path:= $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir:= $(dir $(mkfile_path))

PYTHON=python3.9
PACKAGES=pymongo
PYTHONUSERBASE=resources/python

all: layer-pymongo.zip

layer-pymongo.zip:
	
	mkdir -p $(PYTHONUSERBASE) 
	$(PYTHON) -m pip install $(PACKAGES) -t $(PYTHONUSERBASE)
	wget -O resources/python/global-bundle.pem https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem
	cd resources && zip -r ../layer-pymongo.zip python && cd ..
	rm -rf resources

build-SagemakerEndpointMediatorLambda: 
ifeq ($(OS), Windows_NT)
	copy "$(mkfile_dir)src/SagemakerEndpointMediatorLambda/index.py" "$(ARTIFACTS_DIR)"
else
	cp "$(mkfile_dir)src/SagemakerEndpointMediatorLambda/index.py" "$(ARTIFACTS_DIR)"
endif
	pip install -r "$(mkfile_dir)src/SagemakerEndpointMediatorLambda/requirements.txt" -t "$(ARTIFACTS_DIR)"

build-PostInferenceDocumentDbUpdateLambda: 
ifeq ($(OS), Windows_NT)
	copy "$(mkfile_dir)src/PostInferenceDocumentDbUpdateLambda/index.py" "$(ARTIFACTS_DIR)"
else
	cp "$(mkfile_dir)src/PostInferenceDocumentDbUpdateLambda/index.py" "$(ARTIFACTS_DIR)"
endif
	pip install -r "$(mkfile_dir)src/PostInferenceDocumentDbUpdateLambda/requirements.txt" -t "$(ARTIFACTS_DIR)"

build-NewVideoFileUploadTriggerLambda: 
ifeq ($(OS), Windows_NT)
	copy "$(mkfile_dir)src/NewVideoFileUploadTriggerLambda/index.py" "$(ARTIFACTS_DIR)"
else
	cp "$(mkfile_dir)src/NewVideoFileUploadTriggerLambda/index.py" "$(ARTIFACTS_DIR)"
endif

build-ApiAuthorizerLambda:
ifeq ($(OS), Windows_NT)
	copy "$(mkfile_dir)src/ApiAuthorizer/index.py" "$(ARTIFACTS_DIR)"
else
	cp "$(mkfile_dir)src/ApiAuthorizer/index.py" "$(ARTIFACTS_DIR)"
endif