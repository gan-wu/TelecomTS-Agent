.. This work is licensed under a Creative Commons Attribution 4.0 International License.
.. http://creativecommons.org/licenses/by/4.0
.. Copyright (C) 2019 AT&T Intellectual Property

Release Notes
===============

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <http://keepachangelog.com/>`__
and this project adheres to `Semantic Versioning <http://semver.org/>`__.

[2.5.0] - 2021-06-22
--------------------

* Enhancement to add A1-EI support.
* Upgrade RMR to version 4.5.2.
* Base docker image changed to ubuntu:18.04.
* Upgrade ricxappframe to version 2.0.0.
* Upgrade MDC logging.

[2.4.0] - 2020-12-08
--------------------

* Reference RMR version 4.4.6 via the builder image.

[2.2.0] - 2020-05-28
--------------------

* Add counters of create/update/delete actions on policy types and instances
* Add Prometheus /metrics endpoint to report counter data


[2.1.9] - 2020-05-26
--------------------

* Fix _send_msg method to free allocated RMR message buffers
* Adjust send-message methods to retry only on RMR_ERR_RETRY
* Extend send-message methods to log message state after send
* Use constants from ricxappframe.rmr instead of hardcoded strings
* Upgrade RMR to version 4.0.5
* Upgrade tavern to version 1.2.2
* Extend user guide with southbound API schemas


[2.1.8] - 2020-04-30
--------------------

* Revise Dockerfile to set user as owner of .local dir with a1 package
* Rename console shell start script to run-a1 from run.py
* Extend start script to report webserver listening port
* Add tiny RMR routing table for use in demo and test
* Extend documentation for running a container locally
* Add documentation of start/init parameters to _RmrLoop class
* Add new environment variable USE_FAKE_SDL (`RIC-351 <https://jira.o-ran-sc.org/browse/RIC-351>`_)
* Respond with error if policy type ID differs from ID in object on create
* Upgrade integration tests to use Tavern version 1.0.0


[2.1.7] - 2020-04-28
--------------------

* Upgrade to rmr 4.0.2
* Upgrade integration tests to xapp-frame-go version 0.4.8 which drops NNG
* Extend exception handler to return error details in HTTP response
* Ensure that policy type ID on path matches ID in object
* Add OpenAPI spec to RST documentation


[2.1.6] - 4/7/2020
-------------------

* Switch to rmr 3.6.3
* Switch to using rmr in the ricxappframe


[2.1.5] - 3/19/2020
-------------------

* Switch to python3.8
* Switch to SI95 from NNG (rmr v3 vs rmr v1)
* The switch to SI95 led to a rabbit hole in which we eventually discovered that rmr_send may sometimes block for an arbitrary period of time. Because of this issue, a1's sends are now threaded. Please see the longer comment about this in a1rmr.
* Bump version of py xapp frame (SDL used only) in A1
* Bump version of go xapp frame (0.0.24 -> 0.4.2) in integration tests
* Add some additional logging in A1


[2.1.4] - 3/6/2020
-------------------

* SDL Wrapper was moved into the python xapp framework; use it from there instead.


[2.1.3] - 2/13/2020
-------------------

* This is a pretty big amount of work/changes, however no APIs were changed hence the semver patch
* Switches A1's three test receivers (integration tests) over to golang; this was mostly done to learn the go xapp framework and they are identical in functionality.
* Upgrades the version of rmr in A1 and all integration receivers to 1.13.*
* Uses a much fancier Docker build to reduce the size of a1's image. The python:3.7-alpine image itself is 98MB and A1 is now only ~116MB, so we're done optimizing A1's container size.

[2.1.2] - 1/22/2020
-------------------

* Upgrades from sdl 2.0.2 to 2.0.3
* Integrates an sdl healthcheck into a1's healthcheck


[2.1.1] - 1/14/2020
-------------------

* Upgrades from sdl 1.0.0 to 2.0.2
* Delete a1test_helpers because SDL 2.0.2 provides the mockup we need
* Remove general catch all from A1


[2.1.0] - 1/8/2020
------------------

* Represents a resillent version of 2.0.0 that uses Redis for persistence
* Now relies on SDL and dbaas; SDL is the python interface library to dbaas
* Adds a 503 http code to nearly all http methods, as A1 now depends on an upstream system
* Integration tests have a copy of a dbaas helm chart, however the goal is to simplify that deployment per https://jira.o-ran-sc.org/browse/RIC-45
* Unit tests have a mockup of SDL, however again the goal is to simplify as SDL grows per https://jira.o-ran-sc.org/browse/RIC-44


[2.0.0] - 12/9/2019
-------------------

* Implements new logic around when instances are deleted. See flowcharts in docs/. Basically timeouts now trigger to actually delete instances from a1s database, and these timeouts are configurable.
* Eliminates the barrier to deleting an instance when no xapp evdr replied (via timeouts)
* Add two new ENV variables that control timeouts
* Make unit tests more modular so new workflows can be tested easily
* Fixes the API for ../status to return a richer structure. This is an (albeit tiny) API change.
* Clean up unused items in the integration tests helm chart
* Removed "RMR_RCV_RETRY_INTERVAL" leftovers since this isn't used anymore
* Uses the standard RIC logging library
* Switch the backend routing scheme to using subscription id with constant message types, per request.
* Given the above, policy type ids can be any valid 32bit greater than 0
* Decouple the API between northbound and A1 from A1 with xapps. This is now two seperate OpenAPI files
* Update example for AC Xapp
* Updgrade rmr and rmr-python to utilize new features; lots of cleanups because of that
* Implements a POLICY QUERY feature where A1 listens for queries for a policy type. A1 then responds via multiple RTS messages every policy instance of that policy type (and expects an ACK back from xapps as usual). This feature can be used for xapp recovery etc.


[1.0.4] - 10/24/2019
--------------------

* Only external change here is to healthcheck the rmr thread as part of a1s healthcheck. k8s will now respin a1 if that is failing.
* Refactors (simplifies) how we wait for rmr initialization; it is now called as part of __init__
* Refactors (simplifies) how the thread is actually launched; it is now internal to the object and also a part of __init__
* Cleans up unit testing; a1rmr now exposes a replace_rcv_func; useful for unit testing, harmless if not called otherwise
* Upgrades to rmr-python 1.0.0 for simpler message allocation


[1.0.3] - 10/22/2019
--------------------

* Move database cleanup (e.g., deleting instances based on statuses) into the polling loop
* Rework how unit testing works with the polling loop; prior, exceptions were being thrown silently from the thread but not printed. The polling thread has now been paramaterized with override functions for the purposes of testing
* Make type cleanup more efficient since we know exactly what instances were touched, and it's inefficient to iterate over all instances if they were not
* Bump rmr-python version, and bump rmr version
* Still an item left to do in this work; refactor the thread slightly to tie in a healthcheck with a1s healthcheck. We need k8s to restart a1 if that thread dies too.


[1.0.2] - 10/17/2019
--------------------

* a1 now has a seperate, continuous polling thread, which will enable operations like database cleanup
  (based on ACKs) and external notifications in real time, rather than when the API is invoked
* all rmr send and receive operations are now in this thread
* introduces a thread safe job queue between the two threads
* Not done yet: database cleanups in the thread
* Bump rmr python version
* Clean up some logging


[1.0.1] - 10/15/2019
--------------------

* Moves the "database" access calls to mimick the SDL API, in preparation for moving to SDL
* Does not yet actually use SDL or Redis, but the transition to those will be much shorter after this change.


[1.0.0] - 10/7/2019
-------------------

* Represents v1.0.0 of the A1 API for O-RAN-SC Release A
* Finished here:
  - Implement type DELETE
  - Clean up where policy instance cleanups happen


[0.14.1] - 10/2/2019
--------------------

::

    * Upgrade rmr to 1.9.0
    * Upgrade rmr-python to 0.13.2
    * Use the new helpers module in rmr-python for the rec all functionality
    * Switch rmr mode to a multithreaded mode that continuously reads from rmr and populates an internal queue of messages with a deterministic queue size (2048) which is better behavior for A1
    * Fix a memory leak (python obj is garbage collected but not the underlying C memory allocation)



[0.14.0] - 10/1/2019
--------------------

::

    * Implement instance delete
    * Moves away from the status vector and now aggregates statuses
    * Pop through a1s mailbox "3x as often"; on all 3 kinds of instance GET since all such calls want the latest information
    * Misc cleanups in controller (closures ftw)
    * Add rmr-version.yaml for CICD jobs

[0.13.0] - 9/25/2019
--------------------

::

    * Implement GET all policy type ids
    * Implement GET all policy instance ids for a policy type
    * fix a tiny bug in integration test receiver


[0.12.1] - 9/20/2019
--------------------

::

    * switch to rmr 1.8.1 to pick up a non blocking variant of rmr that deals with bad routing tables (no hanging connections / blocking calls)
    * improve test receiver to behave with this setup
    * add integration test for this case
    * this also switches past 1.5.x, which included another change that altered the behavior of rts; deal with this with a change to a1s helmchart (env: `RMR_SRC_ID`) that causes the sourceid to be set to a1s service name, which was not needed prior
    * improve integration tests overall


[0.12.0] - 9/19/2019
--------------------

::

    * Implement type PUT
    * Implement type GET
    * Remove RIC manifest
    * Read type GET to get schema for instance PUT
    * Remove Utils (no longer needed)
    * lots more tests (unit and integration)

[0.11.0] - 9/17/2019
--------------------

::

    * This is on the road to release 1.0.0. It is not meant to be tested (E2E) as it's own release
    * Implement the Release A spec in the openapi.yaml
    * Rework A1 to follow that spec
    * Remove rmr_mapping now that we use policyid as the mtype to send and a well known mtype for the ACKs
    * Add the delay receiver test to the tavern integration tests
    * Remove unneeded ENV variables from helm charts
    * Switch away from builder images to avoid quicksand; upgrade rmr at our own pace


[0.10.3] - 8/20/2019
--------------------

::

    * Update to later rmr-python
    * Add docs about upgrading rmr
    * remove bombarder since tavern runs apache bench


[0.10.2] - 8/14/2019
--------------------

::

    * Update to later rmr-python

[0.10.1] - 8/9/2019
-------------------

::

    * Greatly reduce the size of A1 docker from 1.25GB to ~278MB.
    * Add a seperate dockerfile for unit testing


[0.10.0] - 7/30/2019
--------------------

::

   * Rename all /ric/ URLs to be consistent with requirements of /A1-P/v2/


[0.9.0] - 7/22/2019
-------------------

::

   * Implement the GET on policies
   * Add a new endpoint for healthcheck. NOTE, it has been decided by oran architecture documents that this policy interface should be named a1-p in all URLS. In a future release the existing URLs will be renamed (existing URLs were not changed in this release).


[0.8.4] - 7/16/2019
-------------------

::

   * Fix the 400, which was in the API, but wasn't actually implemented
   * Update the test fixture manifests to reflect the latest adm control, paves way for next feature coming which is a policy GET



[0.8.3] - 6/18/2019
-------------------

::

   * Use base Docker with NNG version 1.1.1



[0.8.2] - 6/5/2019
------------------

::

   * Upgrade RMR due to a bug that was preventing rmr from init in kubernetes



[0.8.1] - 5/31/2019
-------------------

::

   * Run unit tests as part of docker build



[0.8.0] - 5/28/2019
-------------------

::

   * Convert docs to appropriate format
   * Move rmr string to int mapping to a file



[0.7.2] - 5/24/2019
-------------------

::

   * Use tavern to test the actual running docker container
   * Restructures the integration tests to run as a single tox command
   * Re-ogranizes the README and splits out the Developers guide, which is not needed by users.


[0.7.1] - 5/23/2019
-------------------

::

   * Adds a defense mechanism against A1 getting queue-overflowed with messages A1 doesnt care about; A1 now ignores all incoming messages it's not waiting for, so it's queue size should now always be "tiny", i.e., never exceeding the number of valid requests it's waiting for ACKs back for
   * Adds a test "bombarding" script that tests this


[0.7.0] - 5/22/19
-----------------

::

   * Main purpose of this change is to fix a potential race condition where A1 sends out M1 expecting ACK1, and while waiting for ACK1, sends out M2 expecting ACK2, but gets back ACK2, ACK1. Prior to this change, A1 may have eaten ACK2 and never fufilled the ACK1 request.
   * Fix a bug in the unit tests (found using a fresh container with no RIC manifest!)
   * Fix a (critical) bug in a1rmr due to a rename in the last iteration (RMR_ERR_RMR_RCV_RETRY_INTERVAL)
   * Make unit tests faster by setting envs in tox
   * Move to the now publically available rmr-python
   * Return a 400 if am xapp does not expect a body, but the PUT provides one
   * Adds a new test policy to the example RIC manifest and a new delayed receiver to test the aformentiond race condition


[0.6.0]
-------

::

   * Upgrade to rmr 0.10.0
   * Fix bad api spec RE GET
   * Fix a (big) bug where transactionid wasn't being checked, which wouldn't have worked on sending two policies to the same downstream policy handler


[0.5.1] - 5/13/2019
-------------------

::

   * Rip some testing structures out of here that should have been in rmr (those are now in rmr 0.9.0, upgrade to that)
   * Run Python BLACK for formatting


[0.5.0] - 5/10/2019
-------------------

::

   * Fix a blocking execution bug by moving from rmr's timeout to a non blocking call + retry loop + asyncronous sleep
   * Changes the ENV RMR_RCV_TIMEOUT to RMR_RCV_RETRY_INTERVAL


[0.4.0] - 5/9.2019
------------------

::

   * Update to rmr 0.8.3
   * Change 503 to 504 for the case where downstream does not reply, per recommendation
   * Add a 502 with different reasons if the xapp replies but with a bad/malformed/missing status
   * Make testing much more modular, in anticipating of moving some unit test functionality into rmr itself


[0.3.4] - 5/8/2019
------------------

::

   * Crash immediately if manifest isn't mounted
   * Add unit tests for utils
   * Add missing lic


[0.3.3]
-------

::

   * Upgrade A1 to rmr 0.8.0
   * Go from deb RMR installation to git
   * Remove obnoxious receiver logging


[0.3.2]
-------

::

   * Upgrade A1 to rmr 0.6.0


[0.3.1]
-------

::

   * Add license headers


[0.3.0]
-------

::

   * Introduce RIC Manifest
   * Move some testing functionality into a helper module
   * Read the policyname to rmr type mapping from manifest
   * Do PUT payload validation based on the manifest


[0.2.0]
-------

::

   * Bump rmr python dep version
   * Include a Dockerized test receiver
   * Stencil out the mising GET
   * Update the OpenAPI
   * Include a test docker compose file


[0.1.0]
-------

::

   * Initial Implementation
