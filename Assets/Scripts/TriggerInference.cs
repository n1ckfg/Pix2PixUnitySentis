using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Trigger_informative : MonoBehaviour {

    public Inference_Pix2Pix onnx;

    private void Update() {
        if (Input.GetKeyUp(KeyCode.X)) onnx.DoInference();
    }

}
