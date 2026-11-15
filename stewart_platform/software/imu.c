#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include "lsm6ds3tr-c_reg.h"

#define I2C_BUS   "/dev/i2c-1"
#define IMU_ADDR  0x6A

#define CHECK(call) do { if ((call) != 0) { \
    fprintf(stderr, "%s failed\n", #call); return 1; } } while (0)

/* ---- Linux I2C glue for the ST driver ---- */

static int32_t i2c_xfer(int fd, struct i2c_msg *msgs, int n)
{
    struct i2c_rdwr_ioctl_data x = { .msgs = msgs, .nmsgs = n };
    return ioctl(fd, I2C_RDWR, &x) < 0 ? -1 : 0;
}

static int32_t imu_write(void *h, uint8_t reg, const uint8_t *buf, uint16_t len)
{
    uint8_t tmp[33];
    if (len > 32) return -1;
    tmp[0] = reg;
    memcpy(&tmp[1], buf, len);
    struct i2c_msg m = { IMU_ADDR, 0, len + 1, tmp };
    return i2c_xfer(*(int *)h, &m, 1);
}

static int32_t imu_read(void *h, uint8_t reg, uint8_t *buf, uint16_t len)
{
    struct i2c_msg m[2] = {
        { IMU_ADDR, 0,        1,   &reg },
        { IMU_ADDR, I2C_M_RD, len, buf  },
    };
    return i2c_xfer(*(int *)h, m, 2);
}

/* ---- Sensor setup (accelerometer only) ---- */

static int imu_init(stmdev_ctx_t *ctx)
{
    uint8_t id, rst;

    CHECK(lsm6ds3tr_c_device_id_get(ctx, &id));
    if (id != LSM6DS3TR_C_ID) {
        fprintf(stderr, "bad WHO_AM_I: 0x%02x\n", id);
        return 1;
    }

    CHECK(lsm6ds3tr_c_reset_set(ctx, PROPERTY_ENABLE));
    do { CHECK(lsm6ds3tr_c_reset_get(ctx, &rst)); } while (rst);

    CHECK(lsm6ds3tr_c_block_data_update_set(ctx, PROPERTY_ENABLE));
    CHECK(lsm6ds3tr_c_xl_full_scale_set(ctx, LSM6DS3TR_C_4g));
    CHECK(lsm6ds3tr_c_xl_data_rate_set(ctx, LSM6DS3TR_C_XL_ODR_416Hz));
    return 0;
}

/* ---- Main ---- */

/* imu.c: keep the includes, CHECK macro, i2c_xfer, imu_write and imu_read from before,
   delete main(), and add: */

static int fd = -1;
static stmdev_ctx_t ctx;

int imu_open(void)
{
    fd = open(I2C_BUS, O_RDWR);
    if (fd < 0) return -1;
    ctx = (stmdev_ctx_t){ .write_reg = imu_write, .read_reg = imu_read, .handle = &fd };
    return imu_init(&ctx);          /* the accel-only init from before */
}

int imu_read_mg(float *out)         /* out[3]: ax, ay, az in mg */
{
    int16_t raw[3];
    if (lsm6ds3tr_c_acceleration_raw_get(&ctx, raw)) return -1;
    for (int i = 0; i < 3; i++) out[i] = lsm6ds3tr_c_from_fs4g_to_mg(raw[i]);
    return 0;
}

void imu_close(void) { if (fd >= 0) close(fd); fd = -1; }
