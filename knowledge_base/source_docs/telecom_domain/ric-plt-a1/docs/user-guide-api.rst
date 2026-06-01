.. This work is licensed under a Creative Commons Attribution 4.0 International License.
.. SPDX-License-Identifier: CC-BY-4.0

User Guide and APIs
===================

.. contents::
   :depth: 3
   :local:

This document explains how to communicate with the A1 Mediator.
Information for maintainers of this platform component is in the Developer Guide.

Example Messages
----------------

Send the following JSON to create policy type 20008, which supports instances with
a single integer value:

.. code-block:: yaml

    {
      "name": "tsapolicy",
      "description": "tsa parameters",
      "policy_type_id": 20008,
      "create_schema": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
          "threshold": {
            "type": "integer",
            "default": 0
          }
        },
        "additionalProperties": false
      }
    }


For example, if you put the JSON above into a file called "create.json" you can use
the curl command-line tool to send the request::

.. code::
    
    $ curl -X PUT --header "Content-Type: application/json" --data-raw @create.json "http://localhost/A1-P/v2/policytypes/20008"


Send the following JSON to create an instance of policy type 20008:

.. code-block:: yaml

    {
      "threshold" : 5
    }


For example, you can use the curl command-line tool to send this request::

.. code::

    $ curl -X PUT --header "Content-Type: application/json" --data '{"threshold" : 5}' "http://localhost/A1-P/v2/policytypes/20008/policies/tsapolicy145"


Integrating Xapps with A1
-------------------------

The schema for messages sent by A1 to Xapps is labeled ``downstream_message_schema``
in the Southbound API Specification section below. A1 sends policy instance requests
using message type 20010.

The schemas for messages sent by Xapps to A1 appear in the Southbound API
Specification section below. Xapps must use a message type and content appropriate
for the scenario:

#. When an Xapp receives a CREATE message for a policy instance, the Xapp
   must respond by sending a message of type 20011 to A1. The content is
   defined by schema ``downstream_notification_schema``.  The most convenient
   way is to use RMR's return-to-sender (RTS) feature after setting the
   message type appropriately.
#. Since policy instances can "deprecate" other instances, there are
   times when Xapps need to asynchronously tell A1 that a policy is no
   longer active. Use the same message type and schema as above.
#. Xapps can request A1 to re-send all instances of policy type T using a
   query, message type 20012.  The schema for that message is defined by
   ``policy_query_schema`` (just a body with ``{policy_type_id: ... }``).
   When A1 receives this, A1 will send the Xapp a CREATE message N times,
   where N is the number of policy instances for type T. The Xapp should reply
   normally to each of those as the first item above. That is, after the Xapp
   performs the query, the N CREATE messages sent and the N replies
   are "as normal".  The query just kicks off this process rather than
   an external caller to A1.


Northbound API Specification
----------------------------

This section shows the Open API specification for the A1 Mediator's
northbound interface, which accepts policy type and policy instance requests.
Following are the api and the response::

#. Healthcheck

.. code::

    $ curl -v -X GET "http://localhost/A1-P/v2/healthcheck"

.. code-block:: yaml  

    < HTTP/1.1 200 OK

#. Get all policy types

.. code::

    $ curl -X GET "http://localhost/A1-P/v2/policytypes/"

.. code-block:: yaml

    [20001,5003,21001,21000,21002]


#. Get Policy Type based on policyid

.. code::

    $ curl -s -X GET "http://localhost/A1-P/v2/policytypes/20001" | jq .

.. code-block:: yaml

    {
      "create_schema": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "properties": {
          "additionalProperties": false,
          "blocking_rate": {
            "default": 10,
            "description": "% Connections to block",
            "maximum": 1001,
            "minimum": 1,
            "type": "number"
          },
          "enforce": {
            "default": "true",
            "type": "boolean"
          },
          "window_length": {
            "default": 1,
            "description": "Sliding window length (in minutes)",
            "maximum": 60,
            "minimum": 1,
            "type": "integer"
          }
        },
        "type": "object"
      },
      "description": "various parameters to control admission of dual connection",
      "name": "admission_control_policy_mine",
      "policy_type_id": 20001
    }


#. Get all policy instances for a given policy type

.. code::

    $ curl -s -X GET "http://localhost/A1-P/v2/policytypes/20001/policies/" | jq .

.. code-block:: yaml
  
    [
      "1234",
      "1235"
    ]

#. Get policy instance for a policy id and policy instance id

.. code::

    $ curl -s -X GET "http://localhost/A1-P/v2/policytypes/20001/policies/1234" | jq .

.. code-block:: yaml

    {
      "blocking_rate": 20,
      "enforce": true,
      "trigger_threshold": 20,
      "window_length": 20
    }

#. Create Policy type

.. code::

    $ curl -X PUT "http://localhost/A1-P/v2/policytypes/21003/" -H "Content-Type: application/json" -d @policy_schema_ratecontrol.json


    $ cat policy_schema_ratecontrol.json

.. code-block:: yaml

    {
    "name": "Policy for Rate Control",
      "policy_type_id":21003,
      "description":"This policy is associated with rate control. Entities which support this policy type must accept the following policy inputs (see the payload for more specifics) : class, which represents the class of traffic for which the policy is being enforced",

      "create_schema":{
          "$schema":"http://json-schema.org/draft-07/schema#",
          "type":"object",
          "additionalProperties":false,
          "required":["class"],
          "properties":{
              "class":{
                  "type":"integer",
                  "minimum":1,
                  "maximum":256,
                  "description":"integer id representing class to which we are applying policy"
              },
              "enforce":{
                  "type":"boolean",
                  "description": "Whether to enable or disable enforcement of policy on this class"
              },
              "window_length":{
                  "type":"integer",
                  "minimum":15,
                  "maximum":300,
                  "description":"Sliding window length in seconds"
              },
              "trigger_threshold":{
                  "type":"integer",
                  "minimum":1
              },
              "blocking_rate":{
                  "type":"number",
                  "minimum":0,
                  "maximum":100
              }

          }
      },

      "downstream_schema":{
          "type":"object",
          "additionalProperties":false,
          "required":["policy_type_id", "policy_instance_id", "operation"],
          "properties":{
              "policy_type_id":{
                  "type":"integer",
                  "enum":[21000]
              },
              "policy_instance_id":{
                  "type":"string"
              },
              "operation":{
                  "type":"string",
                  "enum":["CREATE", "UPDATE", "DELETE"]
              },
              "payload":{
                  "$schema":"http://json-schema.org/draft-07/schema#",
                  "type":"object",
                  "additionalProperties":false,
                  "required":["class"],
                  "properties":{
                      "class":{
                          "type":"integer",
                          "minimum":1,
                          "maximum":256,
                          "description":"integer id representing class to which we are applying policy"
                      },
                      "enforce":{
                          "type":"boolean",
                          "description": "Whether to enable or disable enforcement of policy on this class"
                      },
                      "window_length":{
                          "type":"integer",
                          "minimum":15,
                          "maximum":300,
                          "description":"Sliding window length in seconds"
                      },
                      "trigger_threshold":{
                          "type":"integer",
                          "minimum":1
                      },
                      "blocking_rate":{
                          "type":"number",
                          "minimum":0,
                          "maximum":100
                      }


                  }
              }
          }
      },
      "notify_schema":{
          "type":"object",
          "additionalProperties":false,
          "required":["policy_type_id", "policy_instance_id", "handler_id", "status"],
          "properties":{
              "policy_type_id":{
                  "type":"integer",
                  "enum":[21000]
              },
              "policy_instance_id":{
                  "type":"string"
              },
              "handler_id":{
                  "type":"string"
              },
              "status":{
                  "type":"string",
                  "enum":["OK", "ERROR", "DELETED"]
              }
          }
      }
    }




#. Create policy instance

.. code::

    $ curl -X PUT "http://localhost/A1-P/v2/policytypes/21003/policies/1234" -H "Content-Type: application/json" -d @policy_instance_ratecontrol.json
    
    $ cat policy_instance_ratecontrol.json

.. code-block:: yaml

    {
        "class":12,
        "enforce":true,
        "window_length":20,
        "blocking_rate":20,
        "trigger_threshold":10
    }


#. Get policy instance status:
    
.. code::

    $ curl -s -X GET "http://localhost/A1-P/v2/policytypes/21004/policies/1235/status" | jq .

.. code-block:: yaml
  
    {
      "created_at": "0001-01-01T00:00:00.000Z",
      "instance_status": "IN EFFECT"
    }


#. Delete policy type
    
.. code::

    $ curl -s -X DELETE "http://localhost/A1-P/v2/policytypes/21004/"

#. Delete policy instance
    
.. code::

    $ curl -s -X DELETE "http://localhost/A1-P/v2/policytypes/21004/policies/1234/"

#. A1-EI data delivery for a job id:

.. code::

    $ curl -X POST "http://localhost/data-delivery" -H "Content-Type: application/json" -d @a1eidata.json

    $ cat a1eidata.json

.. code-block:: yaml
  
    {
    "job":"100",
    "payload":"payload"
    }



Southbound API Specification
----------------------------

This section shows Open API schemas for the A1 Mediator's southbound interface,
which communicates with Xapps via RMR. A1 sends policy instance requests using
message type 20010. Xapps may send requests to A1 using message types 20011 and
20012.


.. literalinclude:: a1_xapp_contract_openapi.yaml
   :language: yaml
   :linenos:
