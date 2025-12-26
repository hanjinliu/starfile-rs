import tempfile
from pathlib import Path
import starfile_rs.schema.pandas as schema

# `Import/job001/job.star` from https://zenodo.org/records/11068319
JOB_STAR = """
# version 50001

data_job

_rlnJobTypeLabel             relion.importtomo
_rlnJobIsContinue                       0
_rlnJobIsTomo                           0


# version 50001

data_joboptions_values

loop_
_rlnJobOptionVariable #1
_rlnJobOptionValue #2
        Cs        2.7
        Q0        0.1
    angpix      0.675
  do_queue         No
dose_is_per_movie_frame         No
 dose_rate          3
flip_tiltseries_hand        Yes
images_are_motion_corrected         No
        kV        300
mdoc_files mdoc/*.mdoc
min_dedicated         24
movie_files frames/*.mrc
  mtf_file         ""
optics_group_name    optics1
other_args         ""
    prefix         ""
      qsub     sbatch
qsubscript /public/EM/RELION/relion-slurm-gpu-4.0.csh
 queuename    openmpi
tilt_axis_angle         85
"""

class Job(schema.SingleDataModel):
    type_label: str = schema.Field("rlnJobTypeLabel")
    is_continue: int = schema.Field("rlnJobIsContinue")
    is_tomo: int = schema.Field("rlnJobIsTomo")


class JobOptionsValues(schema.LoopDataModel):
    variable: schema.Series[str] = schema.Field("rlnJobOptionVariable")
    value: schema.Series[object] = schema.Field("rlnJobOptionValue")


class JobStarModel(schema.StarModel):
    job: Job = schema.Field("job")
    options: JobOptionsValues = schema.Field("joboptions_values")

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "job_pipeline.star"
        path.write_text(JOB_STAR)
        pipeline = JobStarModel.validate_file(path)

    print(pipeline)
    print(f"{pipeline.job.type_label=!r}")

    parameter_dict = dict(
        zip(
            pipeline.options.variable.tolist(),
            pipeline.options.value.tolist(),
        )
    )
    print("job.star parameters as python dict:")
    print(parameter_dict)

if __name__ == "__main__":
    main()
