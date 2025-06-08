from .yamlCamera import makeCamera

def makeCamera():
    import os
    camera_path = os.path.join(os.path.dirname(__file__), "camera.yaml")
    return makeCamera(camera_path)
