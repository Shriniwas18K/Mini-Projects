<template>
  <header>
    <div><img id="imga" src="./assets/brand.jpeg" /></div>
    PCCOE ART CIRCLE
  </header>

  <div
    class="nv"
    style="
      display: flex;
      width: 100vw;
      flex-flow: column wrap;
      justify-content: center;
      align-items: center;
      align-content: center;
    "
  >
    <button
      class="btn btn-primary"
      style="
        margin: 0.2cm;
        width: 100vw;
        background-color: darkred;
        color: white;
        padding: 0.2cm;
        font-family: Georgia, 'Times New Roman', Times, serif;
      "
      @click.prevent="maincon('home')"
    >
      HOME
    </button>
    <div v-if="submitted == false">
      <button
        class="btn btn-primary"
        style="
          margin: 0.2cm;
          width: 100vw;
          background-color: darkred;
          color: white;
          padding: 0.2cm;
          font-family: Georgia, 'Times New Roman', Times, serif;
        "
        @click.prevent="maincon('registration')"
      >
        REGISTER FOR CLUB
      </button>
    </div>
    <button
      class="btn btn-primary"
      style="
        margin: 0.2cm;
        width: 100vw;
        background-color: darkred;
        color: white;
        padding: 0.2cm;
        font-family: Georgia, 'Times New Roman', Times, serif;
      "
      @click.prevent="maincon('veiw')"
    >
      VEIW REGISTERATIONS
    </button>
  </div>

  <div v-if="main == 'registration'">
    <div class="fr">
      <div v-if="isreg == true" class="form1">
        <p id="hglts">ENROL IN CLUB</p>
        <form @submit.prevent="sub" id="frm">
          <input
            id="inp"
            type="text"
            class="form-control"
            placeholder="enter your name"
            v-model="r.name"
          />
          <input
            id="inp"
            type="text"
            class="form-control"
            placeholder="enter your phoneno."
            v-model="r.phn"
          />
          <input
            id="inp"
            type="text"
            class="form-control"
            placeholder="enter the club name"
            v-model="r.club"
          />
          <button
            style="
              margin: 0.1cm 1cm 0.3cm 1cm;
              padding: 0.1cm;
              color: white;
              background-color: darkred;
              font-family: 'Berkshire Swash', cursive;
            "
          >
            SUBMIT
          </button>
        </form>
      </div>
      <div v-if="submitted == true" class="reg">
        <div id="hglts">You have succefully registered for {{ temp }} club</div>
      </div>
    </div>
  </div>

  <div v-if="main == 'veiw'">
    <div v-if="authenticated == false" class="fr">
      <form
        class="frm"
        id="form1"
        @submit.prevent="auth"
        style="
          display: flex;
          flex-flow: column wrap;
          align-items: center;
          align-content: center;
        "
      >
        <div>ENTER ADMIN USER NAME</div>
        <input type="text" placeholder="" v-model="user" />
        <div>PASSWORD</div>
        <input type="password" placeholder="" v-model="pwd" />
        <button
          style="
            margin: 0.1cm 1cm 0.3cm 1cm;
            padding: 0.1cm;
            color: white;
            background-color: darkred;
            font-family: 'Berkshire Swash', cursive;
          "
        >
          SUBMIT
        </button>
      </form>
    </div>
    <div
      v-if="authenticated == true"
      style="
        display: flex;
        flex-flow: column wrap;
        align-content: center;
        justify-content: center;
      "
    >
      <h2>WELCOME ADMINISTRATOR</h2>
      <table>
        <div v-for="rec in getdata">
          <div style="margin: 0.3cm; border: 1px solid black">
            <tr>
              <td style="border: 1px solid black">{{ rec.name }}</td>
              <td style="border: 1px solid black">{{ rec.phn }}</td>
              <td style="border: 1px solid black">{{ rec.club }}</td>
            </tr>
          </div>
        </div>
      </table>
    </div>
  </div>

  <div v-if="main == 'home'">
    <div class="home">
      <div class="obj">
        <div id="objhd">OBJECTIVE</div>
        <div id="obj">
          To promote cultural activities among students and to provide platform
          to students to show case there various arts / skills / talents.
        </div>
      </div>
      <div class="func">
        <div id="funchd">FUNCTIONS</div>
        <div id="func">
          <p>
            To plan, coordinate and control cultural activities of the students
          </p>
          <p>To guide students to participate in various competitions</p>
          <p>
            To provide proper guidance to students through experts to enhance
            there skills/arts for overall development
          </p>
          <p>
            To arrange for providing essential facilities to the students for
            preparation and practice
          </p>
          <p>
            Monitor and maintain utmost discipline during practice, preparation
            and conduct of the events
          </p>
        </div>
      </div>
      <div id="cassr">
        <div id="hglts" class="cassrhd">HIGHLIGHTS AND ACHIEVEMENTS</div>
        <div id="carouselExampleDark" class="carousel carousel-dark slide">
          <div class="carousel-indicators">
            <button
              type="button"
              data-bs-target="#carouselExampleDark"
              data-bs-slide-to="0"
              class="active"
              aria-current="true"
              aria-label="Slide 1"
            ></button>
            <button
              type="button"
              data-bs-target="#carouselExampleDark"
              data-bs-slide-to="1"
              aria-label="Slide 2"
            ></button>
            <button
              type="button"
              data-bs-target="#carouselExampleDark"
              data-bs-slide-to="2"
              aria-label="Slide 3"
            ></button>
            <button
              type="button"
              data-bs-target="#carouselExampleDark"
              data-bs-slide-to="3"
              aria-label="Slide 4"
            ></button>
            <button
              type="button"
              data-bs-target="#carouselExampleDark"
              data-bs-slide-to="4"
              aria-label="Slide 5"
            ></button>
            <button
              type="button"
              data-bs-target="#carouselExampleDark"
              data-bs-slide-to="5"
              aria-label="Slide 1"
            ></button>
          </div>
          <div class="carousel-inner">
            <div class="carousel-item active" data-bs-interval="10000">
              <img
                src="./assets/firodiya-karandak-2022.jpg"
                class="d-block w-100"
              />
            </div>
            <div class="carousel-item" data-bs-interval="2000">
              <img
                src="./assets/purushottam-karandak-2022-2.jpg"
                class="d-block w-100"
              />
            </div>
            <div class="carousel-item" data-bs-interval="2000">
              <img
                src="./assets/purushottam-karandak-2022-2.jpg"
                class="d-block w-100"
              />
            </div>
            <div class="carousel-item" data-bs-interval="2000">
              <img src="./assets/purushottam.jpeg" class="d-block w-100" />
            </div>
            <div class="carousel-item" data-bs-interval="2000">
              <img src="./assets/vinodottam.jpeg" class="d-block w-100" />
            </div>
          </div>
          <button
            class="carousel-control-prev"
            type="button"
            data-bs-target="#carouselExampleDark"
            data-bs-slide="prev"
          >
            <span class="carousel-control-prev-icon" aria-hidden="true"></span>
            <span class="visually-hidden">Previous</span>
          </button>
          <button
            class="carousel-control-next"
            type="button"
            data-bs-target="#carouselExampleDark"
            data-bs-slide="next"
          >
            <span class="carousel-control-next-icon" aria-hidden="true"></span>
            <span class="visually-hidden">Next</span>
          </button>
        </div>
      </div>
      <div class="swaratarang">
        <div id="hglts" class="swartg">SWARATARNG 2022 TEAM</div>
        <img id="swar" src="./assets/swaratarng.png" />
      </div>
    </div>
  </div>
  <div class="container">
    <footer class="py-3 my-4">
      <ul class="nav justify-content-center border-bottom pb-3 mb-3">
        <li class="nav-item">
          <a
            href="https://artcircle-8e84a.web.app/"
            class="nav-link px-2 text-muted"
            >Home</a
          >
        </li>
        <li class="nav-item">
          <a href="http://www.pccoepune.com/" class="nav-link px-2 text-muted"
            >Pccoe_Official</a
          >
        </li>
        <li class="nav-item">
          <a
            href="https://www.instagram.com/artcircleofficial/"
            class="nav-link px-2 text-muted"
            >Instagram</a
          >
        </li>
      </ul>
      <p class="text-center text-muted">© 2022 PCCOE ART CIRCLE</p>
    </footer>
  </div>
</template>
<script>
import axios from "axios";
export default {
  data() {
    return {
      r: { name: "", phn: "+91", club: "" },
      isreg: false,
      submitted: false,
      temp: "",
      main: "",
      admin: "administrator",
      admpwd: "180205",
      user: "",
      pwd: "",
      authenticated: false,
      getdata: "",
    };
  },
  methods: {
    maincon(temp) {
      this.main = temp;
      if (temp == "registration") {
        this.regist();
      }
    },
    sub() {
      // this link inside post will not work ,make new database in new firebase project and add its link here

      axios
        .post(
          `https://fir-prj-1ced4-default-rtdb.firebaseio.com/posts.json`,
          this.r
        )
        .then((response) => {
          console.log(response);
        });
      /*
      just uncomment this and then data will be actually transferred to database

      we have done this as we are submitting this website to competition nscc web dev
      and we dont want that data should be put by all the world which causes us 
      some king of economic  charges by firebase website
      */
      this.temp = this.r.club;
      this.r.name = "";
      this.r.phn = "+91";
      this.r.club = "";
      this.isreg = false;
      this.submitted = true;
    },
    regist() {
      this.isreg = true;
    },
    auth() {
      if (this.admin == this.user && this.admpwd == this.pwd) {
        this.authenticated = true;
        this.getdt();
        console.log(this.getdata);
      }
    },
    getdt() {
      axios
        .get(`https://fir-prj-1ced4-default-rtdb.firebaseio.com/posts.json`)
        .then((response) => {
          this.formatpdt(response.data);
        })
        .catch((error) => {
          console.log("error");
        });
    },
    formatpdt(rows) {
      this.getdata = [];
      console.log(rows);
      for (let r in rows) {
        this.getdata.push({ ...rows[r], id: r });
      }
    },
  },
};
</script>
<style>
@import url("https://fonts.googleapis.com/css2?family=Berkshire+Swash&display=swap");
</style>
<style>
header {
  font-family: "Berkshire Swash", cursive;
  display: flex;
  flex-flow: column wrap;
  justify-content: center;
}

#hglts {
  font-family: "Berkshire Swash", cursive;
}

#imga {
  height: 14vh;
  width: 14.5vh;
  border: 1px solid blanchedalmond;
  border-radius: 4cm;
  margin-top: 0.2cm;
}

.ctr {
  display: flex;
  align-items: center;
  align-content: center;
}

.fr {
  display: flex;
  align-items: center;
  flex-flow: column wrap;
  margin: 0.2cm;
}

.obj {
  display: flex;
  flex-flow: column wrap;
  align-items: center;
  align-content: center;
  margin: 0.2cm;
  background-color: blanchedalmond;
  padding: 0.2cm;
}

#objhd {
  font-family: "Berkshire Swash", cursive;
  font-size: large;
}

#obj {
  font-family: "Berkshire Swash", cursive;
  font-size: medium;
}

.func {
  display: flex;
  background-color: blanchedalmond;
  flex-flow: column wrap;
  align-items: center;
  align-content: center;
  padding: 0.2cm;
  margin: 0.2cm;
}

#funchd {
  font-family: "Berkshire Swash", cursive;
  font-size: large;
}

#func {
  font-family: "Berkshire Swash", cursive;
  font-size: medium;
  text-align: left;
}

#cassr {
  display: flex;
  align-items: center;
  flex-flow: column wrap;
  align-content: center;
  background-color: blanchedalmond;
  padding: 0.05cm;
}

.cassrhd {
  margin: 0.5cm 0cm 0.5cm 0cm;
}

.form1 {
  display: flex;
  width: fit-content;
  height: max-content;
  align-items: center;
  flex-flow: column wrap;
  background-color: rgb(255, 245, 231);
  border: 1px solid rgb(246, 217, 173);
}

#frm {
  display: flex;
  flex-flow: column wrap;
  align-items: center;
  background-color: blanchedalmond;
}

@media (min-width: 10cm) {
  /*this is for pc specific sizes */
  header {
    font-size: 48px;
    text-align: center;
    background-color: blanchedalmond;
  }

  #hglts {
    font-size: 20px;
  }

  #inp {
    width: 70vw;
    padding: 0.2cm;
    margin: 0.2cm;
  }

  #cassr {
    margin: 0.2cm;
    padding: 1cm;
  }
}

@media (max-width: 10cm) {
  /*this is for mobile specific sizes */
  header {
    font-size: xx-large;
    text-align: center;
    background-color: blanchedalmond;
  }

  #hglts {
    font-size: medium;
  }
  #inp {
    width: 90vw;
  }
}

.reg {
  display: flex;
  padding: 0.3cm;
  border: 1px rgb(236, 217, 189) solid;
  margin: 0.2cm;
  border-radius: 0.1cm;
  background-color: blanchedalmond;
}

.swaratarang {
  margin-top: 0.2cm;
  display: flex;
  flex-flow: column wrap;
  align-items: center;
  width: 100vw;
  background-color: blanchedalmond;
  padding: 0.5cm;
}

.swartg {
  margin: 0.5cm;
}

#swar {
  width: 80vw;
}
</style>
