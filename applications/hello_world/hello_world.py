import os
from holoscan.core import Application
from eason.hello import HelloWorld
class HelloWorldApp(Application):
    def __init__(self):
        super().__init__()
        self.name = "Hello World App"

    def compose(self):
        phrase = self.kwargs("greeting")["phrase"]
        name = self.kwargs("greeting")["name"]
        print(phrase + ',' + name)

if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "hello_world.yaml")
    app = HelloWorldApp()
    app.config(config_file)
    app.run()