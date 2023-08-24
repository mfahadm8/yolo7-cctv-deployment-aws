
mkfile_path:= $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir:= $(dir $(mkfile_path))


build-SagemakerEndpointMediatorLambda: 
ifeq ($(OS), Windows_NT)
	copy "$(mkfile_dir)src/SagemakerEndpointMediatorLambda/index.py" "$(ARTIFACTS_DIR)"
else
	cp "$(mkfile_dir)src/SagemakerEndpointMediatorLambda/index.py" "$(ARTIFACTS_DIR)"
endif
	pip install -r "$(mkfile_dir)src/SagemakerEndpointMediatorLambda/requirements.txt" -t "$(ARTIFACTS_DIR)"


build-ApiAuthorizerLambda:
ifeq ($(OS), Windows_NT)
	copy "$(mkfile_dir)src/ApiAuthorizer/index.py" "$(ARTIFACTS_DIR)"
else
	cp "$(mkfile_dir)src/ApiAuthorizer/index.py" "$(ARTIFACTS_DIR)"
endif