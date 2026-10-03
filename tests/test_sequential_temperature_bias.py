"""Signed spatial errors must stay visible even when their average cancels."""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))

from analyze_sequential_temperature_bias import bias_statistics
from joint_temperature_core import Table


def measured_table():
    x = np.array([[r,0,t,p,1] for p,end in ((100,20),(200,40))
                  for t in range(0,end+1,5) for r in (.001,.02)],dtype=np.float32)
    y = (295.15+2*x[:,2])[:,None]
    return Table(x,y,np.ones(len(x)),x[:,3]).validate()


def test_signed_center_error_survives_opposing_rim_bias():
    table = measured_table()
    prediction = table.y[:,0]+np.where(table.x[:,0]<.005,2.,-2.)
    result = bias_statistics(table,prediction,top=True)
    assert result["full"]["mean_bias_k"]==0
    assert result["center_tail"]["rmse_k"]==2
    for row in result["by_power"].values():
        assert row["center_final_bias_k"]==2
        assert row["radial_tail"]["15_to_25_mm"]["mean_bias_k"]==-2
        assert row["center_tail"]["error_slope_k_per_s"]==0


def test_late_error_uses_each_power_end_and_actual_center_timestamps():
    table = measured_table()
    prediction = table.y[:,0]+.5*table.x[:,2]
    result = bias_statistics(table,prediction,top=True)
    assert result["by_power"]["100"]["tail_start_s"]==15
    assert result["by_power"]["200"]["tail_start_s"]==30
    for row in result["by_power"].values():
        assert row["center_tail"]["error_slope_k_per_s"]==.5
        assert row["center_observations"]["time_s"][-1]==row["end_time_s"]
