import tempfile
from pathlib import Path
import starfile_rs.schema.pandas as schema

# `Select/job018/job_pipeline.star` from https://zenodo.org/records/11068319
JOB_PIPELINE_STAR = """
# version 50001

data_pipeline_general

_rlnPipeLineJobCounter                       2

# version 50001

data_pipeline_processes

loop_
_rlnPipeLineProcessName #1
_rlnPipeLineProcessAlias #2
_rlnPipeLineProcessTypeLabel #3
_rlnPipeLineProcessStatusLabel #4
Select/job018/       None relion.select.interactive    Running

# version 50001

data_pipeline_nodes

loop_
_rlnPipeLineNodeName #1
_rlnPipeLineNodeTypeLabel #2
_rlnPipeLineNodeTypeLabelDepth #3
Class3D/job017/run_it025_optimiser.star OptimiserData.star.relion            1
Select/job018/particles.star ParticleGroupMetadata.star.relion            1

# version 50001

data_pipeline_input_edges

loop_
_rlnPipeLineEdgeFromNode #1
_rlnPipeLineEdgeProcess #2
Class3D/job017/run_it025_optimiser.star Select/job018/

# version 50001

data_pipeline_output_edges

loop_
_rlnPipeLineEdgeProcess #1
_rlnPipeLineEdgeToNode #2
Select/job018/ Select/job018/particles.star
"""

# Use SingleDataModel and LoopDataModel to define the schema of each block
class RelionPipelineGeneral(schema.SingleDataModel):
    count: int = schema.Field("rlnPipeLineJobCounter")

class RelionPipelineProcesses(schema.LoopDataModel):
    name: schema.Series[str] = schema.Field("rlnPipeLineProcessName")
    alias: schema.Series[str] = schema.Field("rlnPipeLineProcessAlias")
    type_label: schema.Series[str] = schema.Field("rlnPipeLineProcessTypeLabel")
    status_label: schema.Series[str] = schema.Field("rlnPipeLineProcessStatusLabel")

class RelionPipelineNodes(schema.LoopDataModel):
    name: schema.Series[str] = schema.Field("rlnPipeLineNodeName")
    type_label: schema.Series[str] = schema.Field("rlnPipeLineNodeTypeLabel")
    type_label_depth: schema.Series[int] = schema.Field("rlnPipeLineNodeTypeLabelDepth")

class RelionPipelineInputEdges(schema.LoopDataModel):
    from_node: schema.Series[str] = schema.Field("rlnPipeLineEdgeFromNode")
    process: schema.Series[str] = schema.Field("rlnPipeLineEdgeProcess")

class RelionPipelineOutputEdges(schema.LoopDataModel):
    process: schema.Series[str] = schema.Field("rlnPipeLineEdgeProcess")
    to_node: schema.Series[str] = schema.Field("rlnPipeLineEdgeToNode")

# Now, define the complete schema for the RELION pipeline STAR file
class RelionPipeline(schema.StarModel):
    """Complete RELION pipeline STAR file schema."""
    general: RelionPipelineGeneral = schema.Field("pipeline_general")
    processes: RelionPipelineProcesses = schema.Field("pipeline_processes")
    nodes: RelionPipelineNodes = schema.Field("pipeline_nodes")
    input_edges: RelionPipelineInputEdges = schema.Field("pipeline_input_edges")
    output_edges: RelionPipelineOutputEdges = schema.Field("pipeline_output_edges")

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "job_pipeline.star"
        path.write_text(JOB_PIPELINE_STAR)
        pipeline = RelionPipeline.validate_file(path)

    # Attributes can safely be accessed
    print(f"{pipeline.general.count=!r}")
    print(f"{pipeline.nodes.type_label_depth.max()=!r}")

    print(pipeline)

    # You can also use a dataclass-like syntax to construct instances
    pipeline = RelionPipeline(
        general=RelionPipelineGeneral(count=5),
        processes=RelionPipelineProcesses(
            name=["proc1", "proc2"],
            alias=["alias1", "alias2"],
            type_label=["type1", "type2"],
            status_label=["status1", "status2"],
        ),
        nodes=RelionPipelineNodes(
            name=["node1", "node2"],
            type_label=["typeA", "typeB"],
            type_label_depth=[1, 2],
        ),
        input_edges=RelionPipelineInputEdges(
            from_node=["nodeX", "nodeY"],
            process=["procX", "procY"],
        ),
        output_edges=RelionPipelineOutputEdges(
            process=["procM", "procN"],
            to_node=["nodeM", "nodeN"],
        )
    )

    print("----- Constructed in Python -----")
    print(pipeline)

if __name__ == "__main__":
    main()
