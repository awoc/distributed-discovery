"""
    This file contains modified code from PM4Py.

    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
"""
from typing import List, Dict, Set, Tuple

import uuid
import xml.etree.ElementTree as et
from xml.dom import minidom
from pm4py.objects.bpmn.obj import BPMN

from distributed_discovery.export.layout import layout_bpmn_graphs, ClusterPosition
from distributed_discovery.objects.message_flow import MessageFlow
from distributed_discovery.util.bpmn import copy_message_bpmn


def write_bpmn(
    file_path: str,
    bpmn_graphs: List[BPMN],
    message_bpmn: Dict[MessageFlow, List[BPMN.BPMNNode]],
    sent_messages: Dict[str, Dict[str, Set[MessageFlow]]],
    endpoints_wh: int = 30,
    task_wh: int = 60,
) -> None:
    """
    Creates a BPMN XML file of a collaboration diagram in the specified file path.

    Parameters
    ----------
    file_path
        The file path of the file.
    bpmn_graphs:
        List of BPMN graphs. One per participant.
    message_bpmn
        A dictionary of activity names and the participant with the corresponding BPMN node.
    sent_messages
        A dictionary of message flows per participant.
    endpoints_wh
        The width and height for a start/end endpoint. Default 30.
    task_wh
        The width and height for a task. Default 60.

    Returns
    -------

    """
    with open(file_path, "wb") as file:
        xml_string = get_xml_string(
            bpmn_graphs,
            copy_message_bpmn(message_bpmn),
            sent_messages,
            endpoints_wh,
            task_wh,
        )
        file.write(xml_string)


def create_collaboration_element(
    collaboration_id: str,
    plane_element: et.Element,
    bpmn_graphs: List[BPMN],
    message_bpmn: Dict[MessageFlow, List[BPMN.BPMNNode]],
    sent_messages: Dict[str, Dict[str, Set[MessageFlow]]],
    lanes_position: Dict[str, ClusterPosition],
) -> et.Element:
    """
    Creates the collaboration XML-element in the BPMN file.

    Parameters
    ----------
    collaboration_id
        The id of the collaboration element.
    plane_element
        The XML-element containing the plane
    bpmn_graphs
        A list of all BPMN graphs.
    message_bpmn
        A dictionary of activity names and the participant with the corresponding BPMN node.
    sent_messages
        A dictionary of message flows per participant.
    lanes_position

    Returns
    -------
    collaboration
        The collaboration XML-element.
    """
    collaboration = et.Element(
        "bpmn:collaboration",
        {"id": collaboration_id},
    )

    for bpmn_graph in bpmn_graphs:
        participant_id = f"id{uuid.uuid4()}"
        name = bpmn_graph.get_name()

        et.SubElement(
            collaboration,
            "bpmn:participant",
            {
                "id": participant_id,
                "name": name,
                "processRef": f"id{bpmn_graph.get_process_id()}",
            },
        )

        shape = et.SubElement(
            plane_element,
            "bpmndi:BPMNShape",
            {
                "bpmnElement": participant_id,
                "id": participant_id + "_gui",
                "isHorizontal": "true",
            },
        )

        position = lanes_position[name]
        et.SubElement(
            shape,
            "omgdc:Bounds",
            {
                "height": str(position.height),
                "width": str(position.width),
                "x": str(position.x),
                "y": str(position.y),
            },
        )

    for participant, entries in sent_messages.items():
        for sender, receivers in entries.items():
            sender_bpmn_nodes = message_bpmn[MessageFlow(participant, sender)]
            receiver_tuples = [
                (node, receiver)
                for receiver in receivers
                for node in message_bpmn[receiver]
            ]

            for sender_bpmn_node, receiver_tuple in zip(
                sender_bpmn_nodes, receiver_tuples
            ):
                receiver_bpmn_node, receiver = receiver_tuple
                message_bpmn[receiver].pop(0)

                flow_id = f"id{uuid.uuid4()}"
                et.SubElement(
                    collaboration,
                    "bpmn:messageFlow",
                    {
                        "id": flow_id,
                        "sourceRef": sender_bpmn_node.get_id(),
                        "targetRef": receiver_bpmn_node.get_id(),
                    },
                )

                edge = et.SubElement(
                    plane_element,
                    "bpmndi:BPMNEdge",
                    {
                        "bpmnElement": flow_id,
                        "id": flow_id + "_gui",
                    },
                )
                et.SubElement(
                    edge,
                    "omgdi:waypoint",
                    {
                        "x": str(sender_bpmn_node.get_x()),
                        "y": str(sender_bpmn_node.get_y()),
                    },
                )
                et.SubElement(
                    edge,
                    "omgdi:waypoint",
                    {
                        "x": str(receiver_bpmn_node.get_x()),
                        "y": str(receiver_bpmn_node.get_y()),
                    },
                )

    return collaboration


# flake8: noqa: C901
def create_process_element(bpmn_graph: BPMN) -> et.Element:
    """
    Creates the process XML-element in the BPMN file.

    Parameters
    ----------
    bpmn_graph
        A list of all BPMN graphs.

    Returns
    -------
    process
        The process XML-element.
    """
    process = et.Element(
        "bpmn:process",
        {
            "id": "id" + bpmn_graph.get_process_id(),
            "isClosed": "false",
            "isExecutable": "false",
            "processType": "None",
        },
    )

    for node in bpmn_graph.get_nodes():
        if isinstance(node, BPMN.StartEvent):
            is_interrupting = (
                "true"
                if node.get_isInterrupting() or isinstance(node, BPMN.MessageStartEvent)
                else "false"
            )
            parallel_multiple = "true" if node.get_parallelMultiple() else "false"
            task = et.SubElement(
                process,
                "bpmn:startEvent",
                {
                    "id": node.get_id(),
                    "isInterrupting": is_interrupting,
                    "name": node.get_name(),
                    "parallelMultiple": parallel_multiple,
                },
            )
            if isinstance(node, BPMN.MessageStartEvent):
                et.SubElement(
                    task, "bpmn:messageEventDefinition", {"id": f"{node.get_id()}_msg"}
                )
        elif isinstance(node, BPMN.EndEvent):
            task = et.SubElement(
                process, "bpmn:endEvent", {"id": node.get_id(), "name": node.get_name()}
            )
            if isinstance(node, BPMN.MessageEndEvent):
                et.SubElement(
                    task, "bpmn:messageEventDefinition", {"id": f"{node.get_id()}_msg"}
                )
        elif isinstance(node, BPMN.IntermediateCatchEvent):
            task = et.SubElement(
                process,
                "bpmn:intermediateCatchEvent",
                {"id": node.get_id(), "name": node.get_name()},
            )
            et.SubElement(
                task, "bpmn:messageEventDefinition", {"id": f"{node.get_id()}_msg"}
            )
        elif isinstance(node, BPMN.IntermediateThrowEvent):
            task = et.SubElement(
                process,
                "bpmn:intermediateThrowEvent",
                {"id": node.get_id(), "name": node.get_name()},
            )
            et.SubElement(
                task, "bpmn:messageEventDefinition", {"id": f"{node.get_id()}_msg"}
            )
        elif isinstance(node, BPMN.BoundaryEvent):
            task = et.SubElement(
                process,
                "bpmn:boundaryEvent",
                {"id": node.get_id(), "name": node.get_name()},
            )
        elif isinstance(node, BPMN.Task):
            task = et.SubElement(
                process, "bpmn:task", {"id": node.get_id(), "name": node.get_name()}
            )
        elif isinstance(node, BPMN.SubProcess):
            task = et.SubElement(
                process,
                "bpmn:subProcess",
                {"id": node.get_id(), "name": node.get_name()},
            )
        elif isinstance(node, BPMN.ExclusiveGateway):
            task = et.SubElement(
                process,
                "bpmn:exclusiveGateway",
                {
                    "id": node.get_id(),
                    "gatewayDirection": node.get_gateway_direction().value.lower(),
                    "name": "",
                },
            )
        elif isinstance(node, BPMN.ParallelGateway):
            task = et.SubElement(
                process,
                "bpmn:parallelGateway",
                {
                    "id": node.get_id(),
                    "gatewayDirection": node.get_gateway_direction().value.lower(),
                    "name": "",
                },
            )
        elif isinstance(node, BPMN.InclusiveGateway):
            task = et.SubElement(
                process,
                "bpmn:inclusiveGateway",
                {
                    "id": node.get_id(),
                    "gatewayDirection": node.get_gateway_direction().value.lower(),
                    "name": "",
                },
            )
        else:
            raise Exception("Unexpected node type.")

        for in_arc in node.get_in_arcs():
            arc_xml = et.SubElement(task, "bpmn:incoming")
            arc_xml.text = "id" + str(in_arc.get_id())

        for out_arc in node.get_out_arcs():
            arc_xml = et.SubElement(task, "bpmn:outgoing")
            arc_xml.text = "id" + str(out_arc.get_id())

    for flow in bpmn_graph.get_flows():
        source = flow.get_source()
        target = flow.get_target()
        et.SubElement(
            process,
            "bpmn:sequenceFlow",
            {
                "id": "id" + str(flow.get_id()),
                "name": flow.get_name(),
                "sourceRef": str(source.get_id()),
                "targetRef": str(target.get_id()),
            },
        )

    return process


def create_diagram_element(
    collaboration_id: str, bpmn_graphs: List[BPMN]
) -> Tuple[et.Element, et.Element]:
    """
    Creates the diagram XML-element in the BPMN file.

    Parameters
    ----------
    collaboration_id
        The id of the collaboration element.
    bpmn_graphs
        A list of all BPMN graphs.

    Returns
    -------
    diagram
        The diagram XML-element.
    plane
        The plane XML-element inside of the diagram element.
    """
    diagram = et.Element(
        "bpmndi:BPMNDiagram",
        {"id": "id" + str(uuid.uuid4()), "name": "diagram"},
    )

    plane = et.SubElement(
        diagram,
        "bpmndi:BPMNPlane",
        {
            "bpmnElement": collaboration_id,
            "id": "id" + str(uuid.uuid4()),
        },
    )

    for bpmn_graph in bpmn_graphs:
        for node in bpmn_graph.get_nodes():
            node_shape = et.SubElement(
                plane,
                "bpmndi:BPMNShape",
                {"bpmnElement": node.get_id(), "id": node.get_id() + "_gui"},
            )
            et.SubElement(
                node_shape,
                "omgdc:Bounds",
                {
                    "height": str(node.get_height()),
                    "width": str(node.get_width()),
                    "x": str(node.get_x()),
                    "y": str(node.get_y()),
                },
            )

        for flow in bpmn_graph.get_flows():
            flow_shape = et.SubElement(
                plane,
                "bpmndi:BPMNEdge",
                {
                    "bpmnElement": "id" + str(flow.get_id()),
                    "id": "id" + str(flow.get_id()) + "_gui",
                },
            )
            for x, y in flow.get_waypoints():
                et.SubElement(flow_shape, "omgdi:waypoint", {"x": str(x), "y": str(y)})

    return diagram, plane


def get_xml_string(
    bpmn_graphs: List[BPMN],
    message_bpmn: Dict[MessageFlow, List[BPMN.BPMNNode]],
    sent_messages: Dict[str, Dict[str, Set[MessageFlow]]],
    endpoints_wh: int,
    task_wh: int,
) -> bytes:
    """
    Creates the BPMN XML file.

    Parameters
    ----------
    bpmn_graphs:
        List of BPMN graphs. One per participant.
    message_bpmn
        A dictionary of activity names and the participant with the corresponding BPMN node.
    sent_messages
        A dictionary of message flows per participant.
    endpoints_wh
        The width and height for a start/end endpoint.
    task_wh
        The width and height for a task.

    Returns
    -------
    xml
        The resulting XML file as byte stream.
    """
    lanes_position = layout_bpmn_graphs(bpmn_graphs, endpoints_wh, task_wh)

    definitions = et.Element("bpmn:definitions")
    definitions.set("xmlns:bpmn", "http://www.omg.org/spec/BPMN/20100524/MODEL")
    definitions.set("xmlns:bpmndi", "http://www.omg.org/spec/BPMN/20100524/DI")
    definitions.set("xmlns:omgdc", "http://www.omg.org/spec/DD/20100524/DC")
    definitions.set("xmlns:omgdi", "http://www.omg.org/spec/DD/20100524/DI")
    definitions.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    definitions.set("targetNamespace", "http://www.signavio.com/bpmn20")
    definitions.set("typeLanguage", "http://www.w3.org/2001/XMLSchema")
    definitions.set("expressionLanguage", "http://www.w3.org/1999/XPath")
    definitions.set("xmlns:xsd", "http://www.w3.org/2001/XMLSchema")

    collaboration_id = f"id{uuid.uuid4()}"

    diagram_element, plane_element = create_diagram_element(
        collaboration_id, bpmn_graphs
    )

    collaboration_element = create_collaboration_element(
        collaboration_id,
        plane_element,
        bpmn_graphs,
        message_bpmn,
        sent_messages,
        lanes_position,
    )

    definitions.append(collaboration_element)

    for bpmn_graph in bpmn_graphs:
        process_element = create_process_element(bpmn_graph)
        definitions.append(process_element)

    definitions.append(diagram_element)

    return minidom.parseString(et.tostring(definitions)).toprettyxml(encoding="utf-8")
