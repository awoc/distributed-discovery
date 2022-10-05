# Distributed Discovery

A process mining Python library based on [PM4Py](https://github.com/pm4py/pm4py-core) to discover inter-organizational 
processes.

## Installation

At least Python version 3.7 is required.

```
pip install distributed-discovery
```

## Usage

```python
import distributed_discovery as dd

log = dd.read_xes("./event-log.xes")
bpmn_diagrams, message_bpmn_nodes, sent_messages = dd.discover_bpmn(log)
dd.view_bpmn(bpmn_diagrams, message_bpmn_nodes, sent_messages)
```
