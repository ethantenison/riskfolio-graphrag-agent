"""Semantic-web export for the redesigned KG induction pipeline.

This module produces a semantically disciplined RDF view over the stabilized
ontology layer and promoted property graph. It keeps ontology and instance
graphs separate, uses SKOS for emergent schemes, OWL for stabilized schema, and
PROV-O for assertion provenance.

Inputs are schema induction outputs and a materialized graph write plan.
Outputs are Turtle serializations plus compact export summaries.

This module does not claim full ontology completeness or SHACL closure.
"""

from __future__ import annotations

from urllib.parse import quote

try:
    from rdflib import Graph, Literal, Namespace
    from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    Graph = None
    Literal = None
    Namespace = None
    OWL = RDF = RDFS = SKOS = XSD = None

from riskfolio_graphrag_agent.kg_models import GraphWritePlan, SchemaInductionResult, SemanticExportResult


def _safe_local(value: str) -> str:
    """Percent-encode a string so it is safe to use as a URI local name.

    RDF URIs must not contain unencoded whitespace, angle brackets or other
    IRI-unsafe characters. Paths from notebook filenames commonly carry spaces.

    Args:
        value: Raw identifier or file-path-derived string.

    Returns:
        Percent-encoded string safe for appending to a namespace URI.
    """
    return quote(value, safe="-._~:@!$&'()*+,;=/#")


class SemanticExportPipeline:
    """Build ontology and instance RDF views for the promoted KG."""

    def run(self, schema_induction: SchemaInductionResult, write_plan: GraphWritePlan) -> SemanticExportResult:
        """Build semantic export artifacts.

        Args:
            schema_induction: Stabilized schema and concept schemes.
            write_plan: Materialized property-graph plan.

        Returns:
            Turtle serializations and export counts.
        """
        if Graph is None or Namespace is None or Literal is None:
            return self._fallback_export(schema_induction, write_plan)

        prov = Namespace("http://www.w3.org/ns/prov#")
        rf = Namespace("https://riskfolio-graphrag.io/kg/v2/")

        ontology_graph = Graph()
        instance_graph = Graph()
        for graph in (ontology_graph, instance_graph):
            graph.bind("rf", rf)
            graph.bind("owl", OWL)
            graph.bind("rdfs", RDFS)
            graph.bind("skos", SKOS)
            graph.bind("prov", prov)
            graph.bind("xsd", XSD)

        for ontology_class in schema_induction.ontology_classes:
            uri = rf[_safe_local(ontology_class.ontology_class_id)]
            ontology_graph.add((uri, RDF.type, OWL.Class))
            ontology_graph.add((uri, RDFS.label, Literal(ontology_class.label)))
            ontology_graph.add((uri, RDFS.comment, Literal(ontology_class.definition)))

        for ontology_property in schema_induction.ontology_properties:
            uri = rf[_safe_local(ontology_property.ontology_property_id)]
            ontology_graph.add((uri, RDF.type, OWL.ObjectProperty))
            ontology_graph.add((uri, RDFS.label, Literal(ontology_property.label)))
            ontology_graph.add((uri, RDFS.comment, Literal(ontology_property.definition)))

        for scheme in schema_induction.concept_schemes:
            scheme_uri = rf[_safe_local(scheme.concept_scheme_id)]
            ontology_graph.add((scheme_uri, RDF.type, SKOS.ConceptScheme))
            ontology_graph.add((scheme_uri, SKOS.prefLabel, Literal(scheme.label)))
            for concept_id in scheme.concept_ids:
                concept_uri = rf[_safe_local(concept_id)]
                ontology_graph.add((concept_uri, RDF.type, SKOS.Concept))
                ontology_graph.add((concept_uri, SKOS.inScheme, scheme_uri))

        for node in write_plan.nodes:
            node_uri = rf[_safe_local(node.node_id)]
            instance_graph.add((node_uri, RDF.type, rf[_safe_local(node.label)]))
            for key, value in node.properties.items():
                predicate = rf[_safe_local(key)]
                if isinstance(value, bool | int | float):
                    instance_graph.add((node_uri, predicate, Literal(value)))
                elif isinstance(value, list):
                    for item in value:
                        instance_graph.add((node_uri, predicate, Literal(item)))
                else:
                    instance_graph.add((node_uri, predicate, Literal(str(value))))
            if node.label == "Assertion":
                instance_graph.add((node_uri, RDF.type, prov.Entity))

        for edge in write_plan.edges:
            source_uri = rf[_safe_local(edge.source_id)]
            target_uri = rf[_safe_local(edge.target_id)]
            predicate_uri = rf[_safe_local(edge.relationship_type)]
            instance_graph.add((source_uri, predicate_uri, target_uri))
            if edge.relationship_type == "SUPPORTED_BY":
                instance_graph.add((source_uri, prov.wasDerivedFrom, target_uri))

        return SemanticExportResult(
            ontology_turtle=ontology_graph.serialize(format="turtle"),
            instances_turtle=instance_graph.serialize(format="turtle"),
            summary={
                "ontology_triples": len(ontology_graph),
                "instance_triples": len(instance_graph),
                "ontology_classes": len(schema_induction.ontology_classes),
                "ontology_properties": len(schema_induction.ontology_properties),
                "materialized_nodes": len(write_plan.nodes),
                "materialized_edges": len(write_plan.edges),
            },
        )

    def _fallback_export(
        self,
        schema_induction: SchemaInductionResult,
        write_plan: GraphWritePlan,
    ) -> SemanticExportResult:
        ontology_lines = ["@prefix rf: <https://riskfolio-graphrag.io/kg/v2/> ."]
        instance_lines = ["@prefix rf: <https://riskfolio-graphrag.io/kg/v2/> ."]

        for ontology_class in schema_induction.ontology_classes:
            ontology_lines.append(
                f'rf:{ontology_class.ontology_class_id} a rf:OntologyClass ; rf:label "{ontology_class.label}" .'
            )
        for ontology_property in schema_induction.ontology_properties:
            ontology_lines.append(
                f'rf:{ontology_property.ontology_property_id} a rf:OntologyProperty ; rf:label "{ontology_property.label}" .'
            )
        for node in write_plan.nodes:
            instance_lines.append(f"rf:{node.node_id} a rf:{node.label} .")
        for edge in write_plan.edges:
            instance_lines.append(f"rf:{edge.source_id} rf:{edge.relationship_type} rf:{edge.target_id} .")

        return SemanticExportResult(
            ontology_turtle="\n".join(ontology_lines),
            instances_turtle="\n".join(instance_lines),
            summary={
                "ontology_triples": len(ontology_lines) - 1,
                "instance_triples": len(instance_lines) - 1,
                "ontology_classes": len(schema_induction.ontology_classes),
                "ontology_properties": len(schema_induction.ontology_properties),
                "materialized_nodes": len(write_plan.nodes),
                "materialized_edges": len(write_plan.edges),
                "export_mode": "fallback",
            },
        )
