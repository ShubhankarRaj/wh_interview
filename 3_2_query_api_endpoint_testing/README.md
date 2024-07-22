This is a module where we are testing a model's `/query` API

## SETUP

```
_> cd 3_2_query_api_endpoint_testing
_> poetry install

# Start the Uvicorn server
_> uvicorn model_service.main:app --reload

```

Test Scenarios Covered

- Invalid input
- Missing input

Besides we can also test for other scenarios like `very-long inputs`, `forbidden characters`, ` numerical input`, etc.

I have configured `ci.yml` in order to run pytests as well as a part of CI-CD.

In order to run the Tests, we can run them using the following command:

```
_> poetry run pytest
```
