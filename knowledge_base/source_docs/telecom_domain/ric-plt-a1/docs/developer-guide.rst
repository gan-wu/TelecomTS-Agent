.. This work is licensed under a Creative Commons Attribution 4.0 International License.
.. http://creativecommons.org/licenses/by/4.0

Developer Guide
===============

.. contents::
   :depth: 3
   :local:

Tech Stack
----------

The A1 Mediator is implemented in Golang.


Running A1 Standalone
---------------------

The A1 container can be run standalone, which means using an in-memory mock
version of SDL and a static route table. The host machine must have the RMR
library and the environment must define the variable `prometheus_multiproc_dir`
with a value like /tmp.  Alternately, use the following command to run A1 as
a Docker container, using a route table mounted as a file from this git
repository and exposing the server's HTTP port on the Docker host::

    docker run -e USE_FAKE_SDL=True -p 10000:10000 -v `pwd`:/opt/route [DOCKER_IMAGE_ID_HERE]

Then test the server with an invocation such as this::

    curl localhost:10000/A1-P/v2/healthcheck


Integration testing
-------------------

This tests A1’s external API with three test receivers. This requires
docker, kubernetes and helm.

Build all the images:

::

    docker build  -t a1:latest .
    cd integration_tests/testxappcode
    docker build -t delayreceiver:latest -f Dockerfile-delay-receiver .
    docker build -t queryreceiver:latest -f Dockerfile-query-receiver .
    docker build -t testreceiver:latest  -f Dockerfile-test-receiver  .


Then, run all the tests from the root (this requires the python packages ``tox``, ``pytest``, and ``tavern``).

::

   tox -c tox-integration.ini

This script:

#. Deploys 3 helm charts (5 containers) into a local kubernetes installation
#. Port forwards a pod ClusterIP to localhost
#. Uses “tavern” to run some tests against the server
#. Barrages the server with Apache bench
#. Tears everything down
