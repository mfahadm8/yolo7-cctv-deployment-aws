
mkfile_path:= $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir:= $(dir $(mkfile_path))


build-SagemakerEndpointMediatorLambda: 
ifeq ($(OS), Windows_NT)
	copy "$(mkfile_dir)src/SagemakerEndpointMediatorLambda" "$(ARTIFACTS_DIR)"
else
	cp "$(mkfile_dir)src/SagemakerEndpointMediatorLambda" "$(ARTIFACTS_DIR)"
endif

build-ApiAuthorizerLambda:
ifeq ($(OS), Windows_NT)
	copy "$(mkfile_dir)src/ApiAuthorizer" "$(ARTIFACTS_DIR)"
else
	cp "$(mkfile_dir)src/ApiAuthorizer" "$(ARTIFACTS_DIR)"
endif