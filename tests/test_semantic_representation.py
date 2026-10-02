import sys,unittest
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from probe_semantic_representation import predict
from analyze_semantic_representation import association
class SemanticRepresentationTests(unittest.TestCase):
    def test_kernel_prediction_matches_primal_ridge_and_ignores_heldout_labels(self):
        x=np.random.default_rng(4).normal(size=(12,7));y=np.linspace(0,1,12)
        train=np.arange(8);test=np.arange(8,12);alpha=.3
        scale=np.mean(np.sum((x[train]-x[train].mean(axis=0))**2,axis=1))
        expected=np.clip(Ridge(alpha=alpha*scale).fit(x[train],y[train]).predict(x[test]),0,1)
        actual=predict(x@x.T,train,test,y,alpha)
        np.testing.assert_allclose(actual,expected,atol=1e-10)
        y[test]=1000
        np.testing.assert_allclose(predict(x@x.T,train,test,y,alpha),actual)
    def test_family_only_relationship_has_no_within_family_outcome_variation(self):
        groups=np.repeat(np.arange(4),6);x=(100*groups+np.tile(np.arange(6),4))[:,None].astype(float)
        y=groups.astype(float);z=np.zeros((24,2));draws=np.array([[0,1,2,3],[3,2,1,0]])
        result=association(x,y,z,groups,draws)
        self.assertGreater(result['pooled_spearman'][0][0],.9)
        self.assertTrue(np.isnan(result['within_family_spearman'][0][0]))
        self.assertEqual(result['within_family_spearman'][1][0]['valid_bootstrap_samples'],0)
if __name__=='__main__':unittest.main()
