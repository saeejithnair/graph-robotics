import json


class Edge:
    def __init__(
        self,
        type: str,
        subject,
        related_object,
        frame_id=None,
        explanation=None,
        description=None,
    ):
        self.type = type
        self.subject = subject
        self.related_object = related_object
        self.frame_id = frame_id
        self.explanation = explanation
        self.description = description

    def verify(self, valid_types, valid_object_names):
        valid_types = [i.lower() for i in valid_types]
        valid_object_names = [i.lower() for i in valid_object_names]
        if not (self.type.lower() in valid_types):
            return False
        if not self.subject.lower() in valid_object_names:
            return False
        if not self.related_object.lower() in valid_object_names:
            return False
        return True

    def json(self):
        if self.description:
            return {
                "type": self.type,
                "subject": self.subject,
                "related object": self.related_object,
                "frame_id": self.frame_id,
                "description": self.description,
            }
        else:
            return {
                "type": self.type,
                "subject": self.subject,
                "related object": self.related_object,
                "frame_id": self.frame_id,
            }

    def __str__(self):
        return json.dumps(
            self.json(),
            indent=2,
        )


def load_edge(json):
    return Edge(
        json["type"],
        json["subject"],
        json["related object"],
        frame_id=json["frame_id"],
        description=json.get("description"),
    )
