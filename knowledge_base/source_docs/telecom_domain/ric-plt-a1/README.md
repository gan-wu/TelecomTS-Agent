# RIC A1 Mediator

The xApp A1 mediator exposes a generic REST API by which xApps can
receive and send northbound messages.  The A1 mediator will take
the payload from such generic REST messages, validate the payload,
and then communicate the payload to the xApp via RMR messaging.

Please see documentation in the docs/ subdirectory.
