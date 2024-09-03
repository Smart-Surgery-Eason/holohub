import os
from argparse import ArgumentParser

from holoscan.core import Application
from holohub.qcap_source import QCAPSourceOp
from holoscan.operators import HolovizOp


class QcapIO(Application):
    def __init__(self,record_type=None):
        super().__init__()
        self.name = "Qcap IO App"
        self.record_type = record_type
        if record_type is not None:
            if record_type not in ("input", "visualizer"):
                raise ValueError("record_type must be either ('input' or 'visualizer')")

    def compose(self):
        # Flow definition
        
        yuan_kwargs = self.kwargs("yuan")
        source = QCAPSourceOp(self, name="yuan", **yuan_kwargs)
        width = yuan_kwargs["width"]
        height = yuan_kwargs["height"]
        is_overlay_enabled = False
        # visualizer_allocator = 
        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=width,
            height=height,
            # enable_render_buffer_input=False,
            # enable_render_buffer_output=False,
            # allocator=visualizer_allocator,
            # cuda_stream_pool=cuda_stream_pool,
            # **self.kwargs("holoviz_overlay" if is_overlay_enabled else "holoviz"),
        )
        self.add_flow(source,visualizer)
        pass
        
if __name__ == "__main__":
    parser = ArgumentParser(description="Endoscopy tool tracking demo application.")
    parser.add_argument(
        "-r",
        "--record_type",
        choices=["none", "input", "visualizer"],
        default="none",
        help="The video stream to record (default: %(default)s).",
    )
    args = parser.parse_args()
    record_type = args.record_type

    config_file = os.path.join(os.path.dirname(__file__), "qcap_io.yaml")
    record_type = args.record_type
    if record_type == "none":
        record_type = None
    
    app = QcapIO(record_type=record_type)
    app.config(config_file)
    app.run()
