using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Barracuda;

public class Inference_Pix2Pix : MonoBehaviour {

    public Camera cam;
    public Transform[] hideTarget;
    public bool camIsMain = true;
    public bool matchMainCamSettings = false;
    public NNModel nnModel;
    public int inferenceResolution = 512;
    public Material targetMtl;
    public string targetMtlProp = "_MainTex";
    public bool displayOutputTexture = true;
    public bool hideRenderLayer = true;
    public bool flipInputX = false;
    public bool flipInputY = true;
    public bool flipOutputX = true;
    public bool flipOutputY = false;
    public bool processInGrayscale = false;
    public bool continuousInteference = false;

    [HideInInspector] public RenderTexture inputRTex;
    [HideInInspector] RenderTexture infRTex;
    [HideInInspector] public RenderTexture outputRTex;
    
    private Model model;
    private IWorker worker;
    private bool ready = true;

    private void Start() {
        if (!camIsMain) {
            cam.enabled = false;
        }
        
        model = ModelLoader.Load(nnModel);
        //worker = WorkerFactory.CreateWorker(model, WorkerFactory.Device.GPU);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);
    }

    private void Update() {
        if (continuousInteference && ready) {
            DoInference();
        }
    }

    public void DoInference() {
        if (ready) {
            StartCoroutine(DoInferenceCR());
        }
    }

    private IEnumerator DoInferenceCR() {
        ready = false;
        if (hideRenderLayer) {
            for (int i=0; i<hideTarget.Length; i++) {
                ChangeRenderLayer(hideTarget[i], LayerMask.NameToLayer("Hidden"), true);
            }
            yield return new WaitForEndOfFrame();
        }

        Screenshot(cam);
        if (infRTex != null) infRTex.Release();
        infRTex = new RenderTexture(inferenceResolution, inferenceResolution, 0, RenderTextureFormat.ARGB32);
        infRTex.enableRandomWrite = true;
        infRTex.Create();
        Graphics.Blit(inputRTex, infRTex);

        // Do inference
        var channelCount = 3; // 1 = grayscale, 3 = rgb, 4 = rgba
        var inputX = new Tensor(infRTex, channelCount);
        Tensor outputY = worker.Execute(inputX).PeekOutput();
        inputX.Dispose(); // Barracuda objects are not GC'd
        float[] outputFloats = outputY.AsFloats();

        // View output as texture
        if (displayOutputTexture) {
            if (outputRTex != null) outputRTex.Release();
            outputRTex = new RenderTexture(infRTex.width, infRTex.height, 0, RenderTextureFormat.ARGB32);
            outputRTex.enableRandomWrite = true;
            outputRTex.Create();
            SetTexFromFloats(outputFloats, processInGrayscale);
            targetMtl.SetTexture(targetMtlProp, outputRTex);
        }
        
        if (hideRenderLayer) {
            for (int i=0; i<hideTarget.Length; i++) {
                ChangeRenderLayer(hideTarget[i], LayerMask.NameToLayer("Default"), true);
            }
        }

        yield return null;

        ready = true;
    }

   private void OnDestroy() {
        worker?.Dispose();

        if (inputRTex != null) inputRTex.Release();
        if (infRTex != null) infRTex.Release();
        if (outputRTex != null) outputRTex.Release();

        inputRTex = null;
        infRTex = null;
        outputRTex = null;
    }

    private void SetTexFromFloats(float[] floatArray, bool isGrayscale) {
        int numPixels = infRTex.width * infRTex.height;
        Debug.Log("Floats: " + floatArray.Length + ", pixels: " + numPixels);

        if (floatArray.Length < numPixels) {
            Debug.LogError("Float array size is smaller than RenderTexture size.");
            return;
        }

        Texture2D tempTex = new Texture2D(infRTex.width, infRTex.height, TextureFormat.ARGB32, false);

        Color[] colors = new Color[numPixels];
        if (isGrayscale) {
            for (int i = 0; i < floatArray.Length; i++) {
                colors[i] = new Color(floatArray[i], floatArray[i], floatArray[i], 1f);
            }
        } else {
            for (int i = 0; i < floatArray.Length; i+=3) {
                int index = i / 3;
                colors[index] = new Color(floatArray[i], floatArray[i+1], floatArray[i+2], 1f);
            }
        }

        tempTex.SetPixels(colors);
        tempTex.Apply();

        float x = 1f;
        float y = 1f;
        if (flipOutputX) x = -1f;
        if (flipOutputY) y = -1f;
        Graphics.Blit(tempTex, outputRTex, new Vector2(x, y), new Vector2(0f, 0f));

        Destroy(tempTex);

        Debug.Log("Floats: " + floatArray.Length + ", pixels: " + numPixels + "\n" +
                  "tempTex: " + tempTex.width + ", " + tempTex.height + "\n" + 
                  "outputRTex: " + outputRTex.width + ", " + outputRTex.height);
    }

    private void Screenshot(Camera cam) {
        if (inputRTex != null) inputRTex.Release();
        inputRTex = new RenderTexture(Screen.width, Screen.height, 0, RenderTextureFormat.ARGB32);
        inputRTex.enableRandomWrite = true;
        inputRTex.Create();

        if (!camIsMain) {
            cam.enabled = true;
            if (matchMainCamSettings) doMatchMainCamSettings(cam);
        }

        cam.targetTexture = inputRTex;
        cam.Render();
        cam.targetTexture = null;

        if (!camIsMain) {
            cam.enabled = false;
        }

        Debug.Log("Screen: " + Screen.width + ", " + Screen.height + "\n" + 
                  "inputRTex: " + inputRTex.width + ", " + inputRTex.height);
    }

    private void doMatchMainCamSettings(Camera cam) {
        cam.nearClipPlane = Camera.main.nearClipPlane;
        cam.farClipPlane = Camera.main.farClipPlane;
        cam.fieldOfView = Camera.main.fieldOfView;
        cam.focalLength = Camera.main.focalLength;
    }

    private Vector3 FindWorldSpaceCoords(Vector2 inputPoint) {
        Ray ray = cam.ScreenPointToRay(inputPoint, Camera.MonoOrStereoscopicEye.Mono);

        RaycastHit hit;

        if (Physics.Raycast(ray, out hit)) {
            return hit.point;
        } else {
            return Vector3.zero;
        }
    }

    List<List<Vector3>> SeparatePointsByDistance(List<Vector3> pointList, float threshold) {
        List<List<Vector3>> separated = new List<List<Vector3>>();
        List<Vector3> currentList = new List<Vector3>();

        for (int i = 0; i < pointList.Count - 1; i++) {
            currentList.Add(pointList[i]);

            float distance = Vector3.Distance(pointList[i], pointList[i + 1]);

            if (distance > threshold) {
                separated.Add(currentList);
                currentList = new List<Vector3>();
            }
        }

        currentList.Add(pointList[pointList.Count - 1]);
        separated.Add(currentList);

        return separated;
    }

    private void ChangeRenderLayer(Transform target, int newLayerIndex, bool changeChildren) {
        target.gameObject.layer = newLayerIndex;

        if (changeChildren) {
            for (int i = 0; i < target.childCount; i++) {
                Transform child = target.GetChild(i);
                ChangeRenderLayer(child, newLayerIndex, true);
            }
        }
    }

}

// https://www.youtube.com/watch?v=ggmArUbRvC4
// https://www.youtube.com/watch?v=R9I9prRUiEo
// https://discussions.unity.com/t/create-texture-from-current-camera-view/86847/2