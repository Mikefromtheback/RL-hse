using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Random = UnityEngine.Random;

public class DoggyAgent : Agent
{
    public ArticulationBody[] legs;
    public float servoSpeed = 150f;
    public ArticulationBody body;
    public GameObject cube;
    public Unity.MLAgentsExamples.GroundContact[] groundContacts;
    public float spawnRadius = 7.5f;
    public float minSpawnDistance = 2.0f;
    public float successDistance = 0.8f;
    public float minBodyHeight = 0.10f;
    public float minUprightDotToFail = -0.5f;
    public float successReward = 10.0f;
    public float fallPenalty = 2.0f;
    public float progressRewardScale = 2.0f;
    public float timePenalty = 0.0005f;
    public float faceTargetRewardScale = 0.002f;
    public float uprightRewardScale = 0.0010f;
    public float angVelPenaltyScale = 0.0002f;
    public float actionEnergyPenaltyScale = 0.00005f;
    private Vector3 defPos;
    private float prevDistToTarget;
    private float ObsDistScale => Mathf.Max(1f, spawnRadius * 3f);
    private const float LinVelScale = 10f;
    private const float AngVelScale = 10f;
    private const float LegVelScale = 20f;
    private const float LegAngVelScale = 20f;

    public override void Initialize()
    {
        defPos = body.transform.position;
        prevDistToTarget = Vector3.Distance(body.transform.position, cube.transform.position);
    }

    public override void OnEpisodeBegin()
    {
        ResetDog();
        ResetTarget();
        prevDistToTarget = Vector3.Distance(body.transform.position, cube.transform.position);
    }

    private void ResetDog()
    {
        Quaternion newRot = Quaternion.Euler(-90f, 0f, Random.Range(0f, 360f));
        body.TeleportRoot(defPos, newRot);
        body.velocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;
        for (int i = 0; i < legs.Length; i++)
            MoveLeg(legs[i], 0f);
    }

    private void ResetTarget()
    {
        for (int tries = 0; tries < 50; tries++)
        {
            var pos = new Vector3(
                Random.Range(-spawnRadius, spawnRadius),
                0.21f,
                Random.Range(-spawnRadius, spawnRadius)
            );
            cube.transform.position = pos;
            float d = Vector3.Distance(body.transform.position, cube.transform.position);
            if (d >= minSpawnDistance)
                break;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 fwd = body.transform.right.normalized;
        Vector3 side = body.transform.forward.normalized;
        Vector3 up = body.transform.up.normalized;
        Vector3 toTarget = cube.transform.position - body.transform.position;
        float toFwd = Vector3.Dot(toTarget, fwd) / ObsDistScale;
        float toSide = Vector3.Dot(toTarget, side) / ObsDistScale;
        float toUp = Vector3.Dot(toTarget, up) / ObsDistScale;
        sensor.AddObservation(Mathf.Clamp(toFwd, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(toSide, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(toUp, -1f, 1f));
        float dist = toTarget.magnitude;
        sensor.AddObservation(Mathf.Clamp(dist / ObsDistScale, 0f, 1f));
        Vector3 toDir = (dist > 1e-6f) ? (toTarget / dist) : fwd;
        float alignment = Vector3.Dot(fwd, toDir);
        sensor.AddObservation(alignment);
        float angle = Vector3.SignedAngle(fwd, toDir, Vector3.up);
        sensor.AddObservation(angle / 180f);
        Vector3 v = body.velocity;
        float vFwd = Vector3.Dot(v, fwd) / LinVelScale;
        float vSide = Vector3.Dot(v, side) / LinVelScale;
        float vUp = Vector3.Dot(v, up) / LinVelScale;
        sensor.AddObservation(Mathf.Clamp(vFwd, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(vSide, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(vUp, -1f, 1f));
        Vector3 w = body.angularVelocity;
        float wFwd = Vector3.Dot(w, fwd) / AngVelScale;
        float wSide = Vector3.Dot(w, side) / AngVelScale;
        float wUp = Vector3.Dot(w, up) / AngVelScale;
        sensor.AddObservation(Mathf.Clamp(wFwd, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(wSide, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(wUp, -1f, 1f));
        float uprightDot = Vector3.Dot(up, Vector3.up);
        sensor.AddObservation(uprightDot);
        foreach (var leg in legs)
        {
            var drive = leg.xDrive;
            float lo = drive.lowerLimit;
            float hi = drive.upperLimit;
            float t = drive.target;
            float target01 = (Mathf.Abs(hi - lo) > 1e-6f) ? Mathf.InverseLerp(lo, hi, t) : 0.5f;
            float targetNorm = target01 * 2f - 1f;
            sensor.AddObservation(Mathf.Clamp(targetNorm, -1f, 1f));
            Vector3 lv = leg.velocity / LegVelScale;
            sensor.AddObservation(Mathf.Clamp(lv.x, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(lv.y, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(lv.z, -1f, 1f));
            Vector3 lw = leg.angularVelocity / LegAngVelScale;
            sensor.AddObservation(Mathf.Clamp(lw.x, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(lw.y, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(lw.z, -1f, 1f));
        }
        foreach (var gc in groundContacts)
            sensor.AddObservation(gc.touchingGround ? 1f : 0f);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var actions = actionBuffers.ContinuousActions;
        int n = Mathf.Min(legs.Length, actions.Length);
        for (int i = 0; i < n; i++)
        {
            float a = Mathf.Clamp(actions[i], -1f, 1f);
            float angle = Mathf.Lerp(legs[i].xDrive.lowerLimit, legs[i].xDrive.upperLimit, (a + 1f) * 0.5f);
            MoveLeg(legs[i], angle);
        }
        Vector3 toTarget = cube.transform.position - body.transform.position;
        float curDist = toTarget.magnitude;
        float delta = prevDistToTarget - curDist;
        delta = Mathf.Clamp(delta, -0.2f, 0.2f);
        AddReward(progressRewardScale * delta);
        prevDistToTarget = curDist;
        AddReward(-timePenalty);
        Vector3 fwd = body.transform.right.normalized;
        Vector3 toDir = (curDist > 1e-6f) ? (toTarget / curDist) : fwd;
        float alignment = Vector3.Dot(fwd, toDir);
        AddReward(faceTargetRewardScale * alignment);
        float uprightDot = Vector3.Dot(body.transform.up.normalized, Vector3.up);
        AddReward(uprightRewardScale * Mathf.Clamp(uprightDot, -1f, 1f));
        AddReward(-angVelPenaltyScale * body.angularVelocity.magnitude);
        float energy = 0f;
        for (int i = 0; i < n; i++)
        {
            float a = Mathf.Clamp(actions[i], -1f, 1f);
            energy += a * a;
        }
        energy /= Mathf.Max(1, n);
        AddReward(-actionEnergyPenaltyScale * energy);

        if (curDist <= successDistance)
        {
            AddReward(successReward);
            EndEpisode();
            return;
        }
        if (body.transform.position.y < minBodyHeight)
        {
            AddReward(-fallPenalty);
            EndEpisode();
            return;
        }

        if (uprightDot < minUprightDotToFail)
        {
            AddReward(-fallPenalty);
            EndEpisode();
            return;
        }
    }

    private void FixedUpdate()
    {
        Debug.DrawRay(body.transform.position, body.transform.right * 2f, Color.red);
        Debug.DrawLine(body.transform.position, cube.transform.position, Color.green);
    }

    private void MoveLeg(ArticulationBody leg, float targetAngle)
    {
        var legComponent = leg.GetComponent<Leg>();
        if (legComponent != null)
            legComponent.MoveLeg(targetAngle, servoSpeed);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;
        float v = Mathf.Sin(Time.time * 5f);
        for (int i = 0; i < ca.Length; i++)
            ca[i] = v;
    }
}
