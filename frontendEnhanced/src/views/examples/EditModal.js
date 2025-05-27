import React, { useState } from "react";
import {
  Modal,
  ModalHeader,
  ModalBody,
  ModalFooter,
  FormGroup,
  Label,
  Input,
  Button,
} from "reactstrap";

const EditModal = ({ isOpen, toggle, documentId, onUpdate }) => {
  const [emb, setEmb] = useState("minilm");
  const [model, setModel] = useState("tinyllama");
  const [isUpdating, setIsUpdating] = useState(false);


  const handleUpdate = async () => {
    setIsUpdating(true);

    try {
        const response = await fetch(`http://localhost:5000/api/documents/${documentId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            emb,
            model,
        }),
        });

        if (!response.ok) throw new Error("Failed to update");

        onUpdate();
    } catch (err) {
        console.error("Update error:", err);
        alert("Update failed");
    } finally {
        setIsUpdating(false);
    }
    };


  return (
    <Modal isOpen={isOpen} toggle={toggle} centered>
      <ModalHeader toggle={toggle}>Reprocess Document</ModalHeader>
      <ModalBody>
        <FormGroup>
          <Label for="embeddingSelect">Embedding Model</Label>
          <Input
            type="select"
            id="embeddingSelect"
            value={emb}
            onChange={(e) => setEmb(e.target.value)}
          >
            <option value="minilm">MiniLM</option>
            <option value="mpnet">MPNet</option>
            <option value="gtr">GTR T5</option>
          </Input>
        </FormGroup>

        <FormGroup>
          <Label for="chatModelSelect">Chat Completion Model</Label>
          <Input
            type="select"
            id="chatModelSelect"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            <option value="tinyllama">TinyLlama</option>
            <option value="zephyr">Zephyr</option>
            <option value="qwen">Qwen</option>
          </Input>
        </FormGroup>
      </ModalBody>
      <ModalFooter>
        <Button
            color="primary"
            onClick={handleUpdate}
            disabled={isUpdating}
            >
            {isUpdating ? (
                <>
                <i className="fas fa-spinner fa-spin mr-2" /> Updating...
                </>
            ) : (
                "Update Document"
            )}
            </Button>
        <Button color="secondary" onClick={toggle}>
          Cancel
        </Button>
      </ModalFooter>
    </Modal>
  );
};

export default EditModal;