/**
 归并排序
 2017-12-12
 */

function mergeSortAnimate(array, sortanimte) {
    if (array.length === 0) {
        return array;
    }
    var i = 0;
    var j = array.length - 1;
    var Sort = function (i, j) {
        console.warn("i=" + i + "j=" + j);
        // 结束条件
        if (i >= j) {
            return;
        }

        sortanimte.activeFragment(i, j);
        //排序
        var mid = Math.floor((i + j) / 2);
        Sort(i, mid);
        Sort(mid + 1, j);
        //合并
        var stepX = i;
        var stepY = mid + 1;

        var InsertFromLeft = function () {
            sortanimte.activeOne(stepX);
            sortanimte.blurOne(stepX);
            ++stepX;
        };

        var InsertFromRight = function () {
            //把stepY逐个挪到左边
            //只能用这种办法了，毕竟演示只支持交换操作
            if (stepX <= mid)
                for (var k = stepY; k > stepX; --k) {
                    sortanimte.exchange(k - 1, k);
                    [array[k - 1], array[k]] = [array[k], array[k - 1]];
                }

            sortanimte.activeOne(stepX);
            sortanimte.blurOne(stepX);
            ++stepX;
            ++stepY;
            ++mid;
        };
        while (stepX <= mid && stepY <= j) {
            if (array[stepX] < array[stepY]) {
                InsertFromLeft();
            } else {
                InsertFromRight();
            }
        }
        while (stepX <= mid) {
            InsertFromLeft();
        }
        while (stepY <= j) {
            InsertFromRight();
        }
    };

    Sort(i, j);

    return array;
}
		